"""The HTTP-entry LangFuse span never records raw exception text.

ECMS-113 established a type-name-only posture for span error output
(``type(e).__name__``, never ``str(e)``): exception messages can carry
user PII, so only the exception class name may reach LangFuse. The
``with_langfuse_tracing`` exception handler must follow the same rule.
Full exception detail stays in the logs, reachable via the tid tag on
the trace.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.asyncio


class _FakeRequest:
    method = "POST"
    headers: dict = {}

    class url:
        path = "/memory/add"


def _capturing_langfuse(captured: dict):
    """Fake Langfuse client whose span records every span.update kwarg."""
    span = MagicMock()
    span.update.side_effect = lambda **kwargs: captured.update(kwargs)

    @contextmanager
    def _cm(**kwargs):
        yield span

    langfuse = MagicMock()
    langfuse.start_as_current_observation.side_effect = _cm
    return langfuse


async def _run_failing_endpoint(captured: dict, exc: Exception):
    with (
        patch("mirix.observability.is_langfuse_enabled", return_value=True),
        patch(
            "mirix.observability.get_langfuse_client",
            return_value=_capturing_langfuse(captured),
        ),
    ):
        # Decorate inside the patch context: with_langfuse_tracing binds
        # the observability accessors at decoration time.
        from mirix.server.rest_api import with_langfuse_tracing

        @with_langfuse_tracing
        async def endpoint():
            raise exc

        with patch(
            "mirix.server.rest_api.get_current_request",
            return_value=_FakeRequest(),
        ):
            with pytest.raises(type(exc)):
                await endpoint()


async def test_error_span_output_is_exception_type_name_only():
    captured: dict = {}
    await _run_failing_endpoint(captured, ValueError("ssn is 123-45-6789"))
    assert captured["level"] == "ERROR"
    assert captured["output"] == {"error_type": "ValueError"}


async def test_raw_exception_message_never_reaches_span():
    captured: dict = {}
    await _run_failing_endpoint(captured, RuntimeError("token=sk-secret bob@acme.com"))
    assert "sk-secret" not in repr(captured)
    assert "bob@acme.com" not in repr(captured)
