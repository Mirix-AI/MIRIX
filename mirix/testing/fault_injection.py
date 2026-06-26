"""Deterministic save-path fault injection for full-stack tests.

VOCABULARY (read this first)
----------------------------
* **site** — *where* a fault is injected. A small fixed set of named points on
  the save path (see ``SITES``): the tool body, the meta-agent LLM loop, the
  sub-agent fan-out, and the two registered-provider boundaries (relational
  write / search read). A hook at each site calls into this module with its
  site name.
* **shape** — *what kind of failure* to simulate (see ``_VALID_SHAPES``): e.g.
  ``transient`` (retryable), ``permanent`` (not), ``conflict`` (duplicate),
  ``attribute_error`` (a pure-Python bug), ``correctable`` (bad tool args the
  LLM can fix), ``llm_chaining_exhausted``. Each shape maps to a specific
  exception type so the EXISTING ``error_policy.classify`` produces the
  expected terminal ``SaveOutcome`` with no classifier change.
* **directive** — one instruction: "at this *site*, raise this *shape*"
  (optionally scoped to a tool/memory-type and a ``fail_attempts`` budget). A
  save carries a list of directives; this module stores them per source.

PROD SAFETY
-----------
This module is the shared kernel for the fault-injection seam, but it is inert
in production: every public entry point returns immediately unless
``settings.fault_injection_enabled`` is True AND the process is not running in a
production app environment. Both gates must pass — defense in depth, so even a
misconfigured flag in prod cannot inject. With the gate closed each call is a
couple of boolean checks and a return: zero behavioral change, no new deps.

WHY A PROD-PATH SEAM (NOT MONKEYPATCH)
--------------------------------------
Full-stack tests run against a real worker in a separate process, so a test can
``monkeypatch`` nothing inside it. The injection points must therefore live in
the code the worker actually runs, gated by an env flag.

HOW IT WORKS
------------
A save request carries its directives in the existing free-form
``source_metadata`` dict under the ``__fault_injection__`` key::

    source_metadata = {
        "__fault_injection__": {
            "faults": [
                {"site": "relational_write", "shape": "transient", "fail_attempts": 1},
                {"site": "tool_body", "shape": "attribute_error", "tool": "episodic"},
            ],
        }
    }

The worker calls :func:`resolve_directives` once it knows the directive bag and
the source id, then each injection hook calls :func:`maybe_raise` with its site
name (and optionally the tool / memory-type it is acting for). Directives are
keyed by ``(source_key, site)`` so concurrent tests using unique source ids
never collide — this is what makes the tests parallel-safe.

Each fire is counted in-process AND emitted on the logger with the
``[FAULT INJECTION]`` prefix so an out-of-process test can grep the service log
and assert the fault actually fired the expected number of times (a passing
test must not be able to mean "the fault never fired"). The terminal
SaveOutcome is read from finalize_source's existing log line — no extra seam.
"""

from __future__ import annotations

import contextvars
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from mirix.errors import (
    CorrectableToolError,
    LLMChainingExhaustedError,
    ProviderConflictError,
    ProviderPermanentError,
    ProviderTransientError,
)
from mirix.log import get_logger
from mirix.settings import settings

logger = get_logger(__name__)

# Prefix on every fire log line so the full-stack tests can grep the aggregated
# service log for fires (and count them) the same way assert_no_error_logs
# greps for ERROR lines. Keep stable — the FST helper matches on it.
LOG_PREFIX = "[FAULT INJECTION]"

# The key under which directives ride inside ``source_metadata``.
DIRECTIVE_KEY = "__fault_injection__"

# Valid hook sites. Kept as a frozenset so a typo'd site in a directive fails
# loudly at resolution time rather than silently never matching.
SITES: frozenset[str] = frozenset(
    {
        "tool_body",  # inside _execute_tool_inner, before the tool body runs
        "llm",  # meta-agent chaining / LLMChainingExhaustedError site
        "llm_request",  # the inner_step LLM call (catchable by overflow recovery)
        "subagent",  # per-sub-agent run inside the fan-out
        "relational_write",  # the registered relational provider's create/update
        "search_read",  # the registered search provider's read path
    }
)

# Sites that sit at a registered-provider boundary (inside that provider's own
# inner-retry tier). For these we raise a status-bearing synthetic exception so
# the provider's existing exception-translation + retry logic handles it exactly
# like a real backend failure — more faithful than raising a typed Provider*
# error directly, and it exercises the translation path end-to-end.
_PROVIDER_SITES: frozenset[str] = frozenset({"relational_write", "search_read"})

# HTTP status code each provider shape masquerades as, so a provider boundary
# that classifies by status routes it like a real backend failure (503 ->
# transient/retried, 400 -> permanent, 409 -> conflict / idempotent no-op).
_PROVIDER_SHAPE_STATUS: Dict[str, int] = {
    "transient": 503,
    "permanent": 400,
    "conflict": 409,
}

# App-env values that mean "production" — the second gate. Any of these
# (case-insensitive) hard-disables injection regardless of the flag.
_PROD_APP_ENVS: frozenset[str] = frozenset({"prod", "production", "prd"})


def _injection_disabled() -> bool:
    """True when fault injection must NOT run. Two independent gates:

    1. ``settings.fault_injection_enabled`` is False (the default), OR
    2. the process is in a production app environment (APP_ENV).

    Belt and suspenders: even if the flag were somehow True in prod, the env
    gate keeps the seam inert.
    """
    if not settings.fault_injection_enabled:
        return True
    if os.environ.get("APP_ENV", "").strip().lower() in _PROD_APP_ENVS:
        return True
    return False


class SyntheticProviderError(Exception):
    """A test-only stand-in for an IPS SDK exception.

    Carries ``status_code`` so ``sdk_exception_translation._exc_status_code``
    reads it and translates to the matching Provider*Error, and so
    ``event_retry.retry_with_backoff`` routes it (retry 5xx, surface 4xx) just
    like the real SDK shapes.
    """

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def _make_exception(shape: str, ctx: str, *, provider_site: bool = False) -> BaseException:
    """Map a fault shape to the exception that drives the documented outcome.

    The shapes deliberately mirror the typed save-path vocabulary so the
    untouched ``error_policy.classify`` yields the right SaveOutcome:

    * ``attribute_error``        -> AttributeError raised in pure agent code
                                    (no provider frame) -> origin-split PERMANENT.
    * ``transient``              -> 503 -> ProviderTransientError -> TRANSIENT.
    * ``permanent``              -> 400 -> ProviderPermanentError -> PERMANENT.
    * ``conflict``               -> 409 -> ProviderConflictError -> idempotent.
    * ``correctable``            -> CorrectableToolError -> bounded LLM re-prompt.
    * ``llm_chaining_exhausted`` -> LLMChainingExhaustedError -> PERMANENT.
    * ``subagent_permanent``     -> ProviderPermanentError from one sub-agent.
    * ``context_overflow``       -> an error carrying the OpenAI "maximum context
                                    length" substring so ``is_context_overflow_error``
                                    matches it and ``inner_step`` runs its
                                    summarize-and-retry recovery (site=llm_request).

    At provider sites the transient/permanent/conflict shapes are raised as a
    status-bearing :class:`SyntheticProviderError` so the registered provider's
    own translation + retry tiers handle them; elsewhere they're raised as the
    typed error directly.
    """
    msg = f"synthetic {shape} fault at {ctx}"
    if shape == "attribute_error":
        return AttributeError(msg)
    if shape == "context_overflow":
        # Carry the OpenAI overflow substring so is_context_overflow_error()
        # matches and inner_step takes its summarize-and-retry recovery branch.
        from mirix.constants import OPENAI_CONTEXT_WINDOW_ERROR_SUBSTRING

        return ValueError(f"{OPENAI_CONTEXT_WINDOW_ERROR_SUBSTRING}: {msg}")
    if shape == "correctable":
        return CorrectableToolError(msg)
    if shape == "llm_chaining_exhausted":
        return LLMChainingExhaustedError(msg)
    if shape == "subagent_permanent":
        return ProviderPermanentError(msg)
    if shape in _PROVIDER_SHAPE_STATUS:
        if provider_site:
            return SyntheticProviderError(msg, status_code=_PROVIDER_SHAPE_STATUS[shape])
        # Off the provider boundary: raise the typed error directly.
        if shape == "transient":
            return ProviderTransientError(msg)
        if shape == "permanent":
            return ProviderPermanentError(msg)
        return ProviderConflictError(msg)
    # Unreachable: shapes are validated at resolution time.
    raise ValueError(f"unknown fault shape: {shape!r}")


# Validated at resolution time so a typo surfaces immediately.
_VALID_SHAPES: frozenset[str] = frozenset(
    {
        "attribute_error",
        "transient",
        "permanent",
        "conflict",
        "correctable",
        "llm_chaining_exhausted",
        "subagent_permanent",
        "context_overflow",
    }
)


@dataclass
class FaultDirective:
    """One fault to inject at one site for one source.

    ``fail_attempts`` controls recover-vs-sustained behaviour: when set, the
    directive fires for the first N calls at this site then stops (the inner
    retry tier recovers on attempt N+1). When None, every call fires (sustained
    failure). ``tool`` optionally scopes the directive to a named tool /
    memory-type so partial fan-out tests can fail exactly one sub-agent.
    """

    site: str
    shape: str
    fail_attempts: Optional[int] = None
    tool: Optional[str] = None
    fired: int = field(default=0)

    def matches(self, site: str, tool: Optional[str]) -> bool:
        if self.site != site:
            return False
        if self.tool is not None and self.tool != tool:
            return False
        return True

    def should_fire(self) -> bool:
        """Whether this directive should fire on the current call, accounting
        for the fail_attempts budget. Does not mutate state."""
        if self.fail_attempts is None:
            return True
        return self.fired < self.fail_attempts


# Process-wide registry. The MIRIX worker runs the save path as coroutines on a
# single asyncio event loop (no threads / executors on this path), so a plain
# dict/set mutation between two await-free statements is atomic — no lock is
# needed. (A threading.Lock would be worse than useless here: held across a
# future stray ``await`` it would deadlock the event loop.)
_directives: Dict[str, List[FaultDirective]] = {}
_fire_counts: Dict[tuple[str, str], int] = {}
# Source keys we've already resolved, so resolve_directives is idempotent even
# when called from multiple hook sites for the same source.
_resolved: set[str] = set()


# The save currently in flight on this async context. Set by the meta-agent
# step() when it resolves directives, read by hook sites that lack a direct
# memory_source_id handle (notably the registered-provider boundary, which
# writes many tables for one save and never receives the source id as an
# argument). ContextVars are copied into child tasks at asyncio.gather time, so
# sub-agent provider writes inherit the active source automatically.
_active_source: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "fault_injection_active_source", default=None
)


def set_active_source(source_key: Optional[str]) -> Optional[contextvars.Token]:
    """Mark ``source_key`` as the save in flight on this async context.

    No-op (returns None) when injection is disabled. Returns a token the caller
    may pass to :func:`reset_active_source` to restore the previous value.
    """
    if _injection_disabled():
        return None
    return _active_source.set(source_key)


def get_active_source() -> Optional[str]:
    """The source_key for the save in flight on this async context, if any."""
    if _injection_disabled():
        return None
    return _active_source.get()


def reset_active_source(token: Optional[contextvars.Token]) -> None:
    if token is not None:
        try:
            _active_source.reset(token)
        except (ValueError, LookupError):  # pragma: no cover - cross-context reset
            pass


def reset() -> None:
    """Clear all registry state. For test isolation only."""
    _directives.clear()
    _fire_counts.clear()
    _resolved.clear()
    _active_source.set(None)


def resolve_directives(source_key: str, source_metadata: Optional[dict]) -> None:
    """Register fault directives for ``source_key`` from a source_metadata bag.

    No-op when injection is disabled, when there is no directive bag, or when
    this source was already resolved (idempotent). Raises ValueError on an
    unknown site or shape so a malformed directive fails loudly rather than
    silently never firing.
    """
    if _injection_disabled():
        return
    if not source_key or not source_metadata:
        return
    bag = source_metadata.get(DIRECTIVE_KEY)
    if not bag:
        return

    if source_key in _resolved:
        return
    _resolved.add(source_key)

    parsed: List[FaultDirective] = []
    for raw in bag.get("faults", []):
        site = raw.get("site")
        shape = raw.get("shape")
        if site not in SITES:
            raise ValueError(f"fault-injection: unknown site {site!r} (valid: {sorted(SITES)})")
        if shape not in _VALID_SHAPES:
            raise ValueError(f"fault-injection: unknown shape {shape!r} (valid: {sorted(_VALID_SHAPES)})")
        parsed.append(
            FaultDirective(
                site=site,
                shape=shape,
                fail_attempts=raw.get("fail_attempts"),
                tool=raw.get("tool"),
            )
        )

    _directives.setdefault(source_key, []).extend(parsed)
    logger.info("%s resolved %d directive(s) for source=%s", LOG_PREFIX, len(parsed), source_key)


def _take_matching_directive(site: str, source_key: Optional[str], tool: Optional[str]) -> Optional[FaultDirective]:
    """Find a matching directive for (source_key, site, tool), and if it should
    fire on this call, increment its budget + the fire counter and return it.
    Returns None when injection is disabled or nothing matches/fires."""
    # Diagnostic for the recurring "directive resolved but never fired" question:
    # name the reason maybe_raise/next_fault no-ops, so a no-fire is debuggable
    # without guessing (disabled? no source on this call? no directive registered
    # for this source? site/tool mismatch? budget spent?). DEBUG-level: silent in
    # normal ops, visible under the FST's DEBUG logging.
    if _injection_disabled():
        return None
    if not source_key:
        logger.debug("%s no-fire site=%s reason=no_active_source", LOG_PREFIX, site)
        return None
    directives = _directives.get(source_key)
    if not directives:
        logger.debug(
            "%s no-fire site=%s source=%s reason=no_directives_for_source",
            LOG_PREFIX,
            site,
            source_key,
        )
        return None
    directive = next((d for d in directives if d.matches(site, tool)), None)
    if directive is None:
        logger.debug(
            "%s no-fire site=%s source=%s tool=%s reason=no_matching_directive " "(registered sites=%s)",
            LOG_PREFIX,
            site,
            source_key,
            tool,
            [d.site for d in directives],
        )
        return None
    if not directive.should_fire():
        logger.debug(
            "%s no-fire site=%s source=%s reason=budget_spent (fired=%d fail_attempts=%s)",
            LOG_PREFIX,
            site,
            source_key,
            directive.fired,
            directive.fail_attempts,
        )
        return None
    directive.fired += 1
    key = (source_key, site)
    _fire_counts[key] = _fire_counts.get(key, 0) + 1
    _log_fire(directive.shape, site, source_key, tool)
    return directive


def _log_fire(shape: str, site: str, source_key: str, tool: Optional[str]) -> None:
    logger.warning(
        "%s fired shape=%s site=%s source=%s tool=%s",
        LOG_PREFIX,
        shape,
        site,
        source_key,
        tool or "-",
    )


def maybe_raise(site: str, *, source_key: Optional[str] = None, tool: Optional[str] = None) -> None:
    """Raise the configured fault for ``site`` if one matches; otherwise no-op.

    Call sites invoke this unconditionally — both the enabled flag and the
    prod-env gate are checked inside, so there is no guard to repeat at the
    call site.
    """
    directive = _take_matching_directive(site, source_key, tool)
    if directive is None:
        return
    ctx = f"site={site} source={source_key}" + (f" tool={tool}" if tool else "")
    raise _make_exception(directive.shape, ctx, provider_site=site in _PROVIDER_SITES)


def next_fault(site: str, *, source_key: Optional[str] = None, tool: Optional[str] = None) -> Optional[str]:
    """Like :func:`maybe_raise`, but RETURN the matched shape (and record the
    fire) instead of raising. For hook sites that must raise a native exception
    shape of their own — e.g. a search-read boundary that retries ``httpx``
    errors, so the injected fault has to be an ``httpx`` exception for that
    boundary's existing retry+translation tier to handle it.

    Returns None (no fire) when injection is disabled, no directive matches, or
    the fail_attempts budget is spent.
    """
    directive = _take_matching_directive(site, source_key, tool)
    return directive.shape if directive is not None else None


def fire_count(source_key: str, site: str) -> int:
    """How many times a fault fired at (source_key, site) in this process."""
    return _fire_counts.get((source_key, site), 0)
