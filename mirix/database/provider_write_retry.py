"""Conflict / transient predicates for registered-provider writes.

Each registered provider catches its SDK exceptions at the boundary and
re-raises one of the MIRIX-owned types (`ProviderTransientError`,
`ProviderPermanentError`, `ProviderConflictError`, `ProviderNotFoundError`).
Classification here is **pure isinstance** — no structural detection, no
string-matching, no SDK class imports in core.

There used to be a `retry_transient` helper in this module that wrapped
provider calls in an extra retry loop. It was deleted: every registered
provider already has its own inner-retry tier (e.g.
`event_retry.retry_with_backoff` for IPS-R, `_post_json` for IPS-Search,
the SQLAlchemy `retry_db_operation`/`transaction_retry` decorators), and
all of them tag the propagated exception with the inner-exhausted marker
on exhaustion so the whole-step policy doesn't re-retry. Wrapping a
provider call in a second retry loop only stacked the two budgets
without adding recovery — provider handles its own retries; managers
just classify the result.

The two predicates below survive because the manager error-classification
branches use them to route exceptions:

* **Conflict** — unique-constraint violation. Caller treats as no-op
  (idempotent re-write). `isinstance(exc, ProviderConflictError)`.
* **Transient** — provider exhausted its retry budget. Caller logs +
  emits a skip-span and continues. `isinstance(exc, ProviderTransientError)`.
* **Permanent** — anything else (auth, unknown entity, malformed
  request). Caller re-raises so the policy layer classifies.
"""

from mirix.errors import (
    ProviderConflictError,
    ProviderPermanentError,
    ProviderTransientError,
)


def is_conflict(exc: BaseException) -> bool:
    """True if the exception is a duplicate-key / unique-constraint violation.

    Used by `source_message_manager` (L1 dedup), `user_manager`,
    `client_manager`, `memory_citation_manager` (L3 dedup). Each treats
    a conflict as an idempotent no-op (`continue` / `if not is_conflict(exc): raise`).
    """
    return isinstance(exc, ProviderConflictError)


def is_transient(exc: BaseException) -> bool:
    """True if the exception escaped a provider's inner-retry tier after
    its budget exhausted. Caller decides whether to log+skip or surface.

    Conflict and Permanent shapes always short-circuit to False; only
    ProviderTransientError is retryable.
    """
    if isinstance(exc, (ProviderPermanentError, ProviderConflictError)):
        return False
    return isinstance(exc, ProviderTransientError)
