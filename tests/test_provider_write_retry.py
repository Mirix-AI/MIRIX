"""Unit tests for the MIRIX provider write classification predicates.

After translation is wired in at the ECMS provider boundary, MIRIX core
classifies via pure isinstance against the MIRIX-owned types. No
structural detection lives here.

The provider's own inner-retry tier handles retries (event_retry.retry_with_backoff
for IPS-R, _post_json for IPS-Search, the SQLAlchemy retry decorators
for ORM ops). MIRIX no longer wraps provider calls in its own retry loop —
managers just call the provider and classify whatever exception escapes.

Tests for legacy structural detection (status_code parsing, `uq_*`
constraint-name parsing, `BadRequestError` ambiguity) live in
`app/tests/unit/ipsr/test_sdk_exception_translation.py` against the
translator that produces these types.

Run:
    pytest tests/test_provider_write_retry.py -v
"""

import asyncio

from mirix.database.provider_write_retry import (
    is_conflict,
    is_transient,
)
from mirix.errors import (
    ProviderConflictError,
    ProviderPermanentError,
    ProviderTransientError,
)


class TestIsConflict:
    """is_conflict is pure isinstance against ProviderConflictError."""

    def test_provider_conflict_error_is_conflict(self):
        assert is_conflict(ProviderConflictError("uq_email")) is True

    def test_provider_permanent_error_is_not_conflict(self):
        assert is_conflict(ProviderPermanentError("400 bad shape")) is False

    def test_provider_transient_error_is_not_conflict(self):
        assert is_conflict(ProviderTransientError("503")) is False

    def test_arbitrary_exception_is_not_conflict(self):
        assert is_conflict(Exception("unique constraint violation")) is False

    def test_arbitrary_exception_with_409_is_not_conflict(self):
        """Structural-409 detection is gone. The provider boundary
        translates 409 to ProviderConflictError; raw status codes
        no longer match."""

        class _StatusError(Exception):
            status_code = 409

        assert is_conflict(_StatusError()) is False


class TestIsTransient:
    """is_transient is pure isinstance against ProviderTransientError."""

    def test_provider_transient_error_is_transient(self):
        assert is_transient(ProviderTransientError("503 from IPS-R")) is True

    def test_provider_permanent_error_is_not_transient(self):
        assert is_transient(ProviderPermanentError("400")) is False

    def test_provider_conflict_error_is_not_transient(self):
        """Conflict short-circuits to False — must not be retried."""
        assert is_transient(ProviderConflictError("uq_email")) is False

    def test_arbitrary_status_503_is_not_transient(self):
        """Structural detection is gone."""

        class _StatusError(Exception):
            status_code = 503

        assert is_transient(_StatusError()) is False

    def test_asyncio_timeout_is_not_transient(self):
        """Raw asyncio.TimeoutError → transient is gone — the provider
        boundary translates timeouts into ProviderTransientError."""
        assert is_transient(asyncio.TimeoutError()) is False

    def test_provider_conflict_takes_precedence_over_transient_check(self):
        """A ProviderConflictError must NOT be retried even if its message
        contains words that would otherwise look transient."""
        exc = ProviderConflictError("Timeout while writing duplicate row")
        assert is_conflict(exc) is True
        assert is_transient(exc) is False
