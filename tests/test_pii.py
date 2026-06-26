"""Tests for mirix.pii.

Covers the async :func:`log_error_strip_pii` helper end-to-end:
template formatting, ispy-pii masking of the exception message, and
the frames-only traceback shape.
"""

import logging

import httpx
import pytest

from mirix.pii import REDACTED_PLACEHOLDER, log_error_strip_pii, mask_structure

# Project pytest.ini does not enable asyncio_mode=auto; mark the whole
# module so every async test in this file runs under pytest-asyncio.
pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _enable_pii(monkeypatch):
    monkeypatch.setenv("MIRIX_ISPY_PII_ENABLED", "true")
    monkeypatch.setenv("MIRIX_ISPY_PII_ENDPOINT", "http://ispy.test/v2/analyze")
    monkeypatch.setenv("MIRIX_ISPY_PII_TIMEOUT_MS", "200")
    yield


@pytest.fixture(autouse=True)
def _reset_module_client():
    """The module caches an httpx.AsyncClient at module scope. Reset
    between tests so the MockTransport patched by one test never bleeds
    into the next, and so the singleton can never be observed mid-test
    bound to a transport that has already been torn down.

    Resetting here is also the seam tests use to install a fake
    transport: each test assigns ``mirix.pii._async_client`` to an
    AsyncClient backed by ``httpx.MockTransport`` *before* invoking the
    helper, and the autouse cleanup wipes it afterward.
    """
    import mirix.pii as _pii

    _pii._async_client = None
    yield
    _pii._async_client = None


def _install_mock_async_client(handler):
    """Install an httpx.AsyncClient backed by MockTransport as the
    module-level singleton. The helper's _get_async_client() returns
    the already-set instance, so this is the seam for stubbing.
    """
    import mirix.pii as _pii

    _pii._async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _raise_with_user_msg(user_input):
    """Source line of this function does NOT contain the runtime value,
    so format_tb's source rendering of this frame is PII-free."""
    raise ValueError(user_input)


async def test_log_error_strip_pii_includes_template_masked_msg_and_traceback(caplog):
    """The helper renders <fmt % args> error_type=<type> msg=<masked> + tb,
    and the traceback ends in the bare exception type (no message)."""

    def handler(request):
        return httpx.Response(200, json={"redactedText": "ssn ***-**-6789"})

    _install_mock_async_client(handler)
    test_logger = logging.getLogger("test_pii_async_happy")
    caplog.set_level(logging.ERROR, logger=test_logger.name)

    runtime_pii = "user " + "ssn=" + "1" + "23-45-6789"
    try:
        _raise_with_user_msg(runtime_pii)
    except ValueError as e:
        await log_error_strip_pii(test_logger, "Failed to X: user_id=%s", "u_456", exc=e)

    rec = caplog.records[-1]
    msg = rec.getMessage()
    assert "Failed to X: user_id=u_456" in msg
    assert "error_type=ValueError" in msg
    assert "msg=ssn ***-**-6789" in msg
    assert "Traceback (most recent call last):" in msg
    assert "_raise_with_user_msg" in msg
    assert msg.rstrip().endswith("ValueError")
    assert "123-45-6789" not in msg


async def test_log_error_strip_pii_uses_placeholder_when_mask_fails():
    def handler(request):
        return httpx.Response(500, text="boom")

    _install_mock_async_client(handler)
    test_logger = logging.getLogger("test_pii_async_placeholder")
    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    test_logger.addHandler(_Capture())
    test_logger.setLevel(logging.ERROR)

    runtime_pii = "user " + "ssn=" + "1" + "23-45-6789"
    try:
        _raise_with_user_msg(runtime_pii)
    except ValueError as e:
        await log_error_strip_pii(test_logger, "X failed", exc=e)

    msg = records[-1].getMessage()
    assert REDACTED_PLACEHOLDER in msg
    assert "23-45-6789" not in msg


async def test_log_error_strip_pii_passthrough_when_disabled(monkeypatch, caplog):
    """Kill switch off → mask is a passthrough → str(e) reaches log
    verbatim. The traceback portion still strips the exception-message
    line via safe_traceback."""
    monkeypatch.setenv("MIRIX_ISPY_PII_ENABLED", "false")

    test_logger = logging.getLogger("test_pii_async_disabled")
    caplog.set_level(logging.ERROR, logger=test_logger.name)

    runtime_pii = "ssn " + "1" + "23-45-6789"
    try:
        _raise_with_user_msg(runtime_pii)
    except ValueError as e:
        await log_error_strip_pii(test_logger, "X failed", exc=e)

    msg = caplog.records[-1].getMessage()
    assert "msg=ssn 123-45-6789" in msg
    assert msg.rstrip().endswith("ValueError")


async def test_log_error_strip_pii_supports_warning_level():
    test_logger = logging.getLogger("test_pii_async_warning")
    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    test_logger.addHandler(_Capture())
    test_logger.setLevel(logging.WARNING)

    def handler(request):
        return httpx.Response(200, json={"redactedText": "ok"})

    _install_mock_async_client(handler)

    try:
        _raise_with_user_msg("oops")
    except ValueError as e:
        await log_error_strip_pii(test_logger, "X failed", exc=e, level=logging.WARNING)

    assert records[-1].levelno == logging.WARNING


async def test_log_error_strip_pii_renders_chained_exceptions(caplog):
    def handler(request):
        return httpx.Response(200, json={"redactedText": "outer-msg"})

    _install_mock_async_client(handler)
    test_logger = logging.getLogger("test_pii_async_chained")
    caplog.set_level(logging.ERROR, logger=test_logger.name)

    inner_pii = "inner-" + "111-22-3333"
    outer_pii = "outer-" + "999-88-7777"
    try:
        try:
            raise ValueError(inner_pii)
        except ValueError as inner:
            raise RuntimeError(outer_pii) from inner
    except RuntimeError as e:
        await log_error_strip_pii(test_logger, "X failed", exc=e)

    msg = caplog.records[-1].getMessage()
    assert "RuntimeError" in msg
    assert "ValueError" in msg
    assert "111-22-3333" not in msg
    assert "999-88-7777" not in msg


async def test_log_error_strip_pii_swallows_arbitrary_network_errors():
    """If the httpx call raises (timeout, connection reset, etc.), we
    fall back to the placeholder rather than re-entering the caller's
    exception handler."""

    def handler(request):
        raise OSError("connection reset by peer")

    _install_mock_async_client(handler)
    test_logger = logging.getLogger("test_pii_async_oserror")
    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    test_logger.addHandler(_Capture())
    test_logger.setLevel(logging.ERROR)

    runtime_pii = "ssn " + "1" + "23-45-6789"
    try:
        _raise_with_user_msg(runtime_pii)
    except ValueError as e:
        await log_error_strip_pii(test_logger, "X failed", exc=e)

    msg = records[-1].getMessage()
    assert REDACTED_PLACEHOLDER in msg
    # Runtime PII (built at call time, not on the source line) does
    # not appear anywhere in the formatted log.
    assert "123-45-6789" not in msg


async def test_log_error_strip_pii_survives_pathological_exception_str():
    """If exc.__str__() itself raises, the helper must not re-enter the
    caller's exception handler. It falls back to the
    `(mask helper failed)` shape and logs only the safe template."""

    class _BadException(Exception):
        def __str__(self):
            raise RuntimeError("__str__ bombed")

    test_logger = logging.getLogger("test_pii_async_bad_str")
    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    test_logger.addHandler(_Capture())
    test_logger.setLevel(logging.ERROR)

    try:
        raise _BadException()
    except _BadException as e:
        # Must NOT raise. If we get here, the helper held its contract.
        await log_error_strip_pii(test_logger, "X failed: id=%s", "abc", exc=e)

    # Last-ditch fallback should have logged the safe template.
    msg = records[-1].getMessage()
    assert "X failed: id=abc" in msg
    assert "mask helper failed" in msg


async def test_log_error_strip_pii_survives_cyclic_exception_chain():
    """An exception whose __cause__/__context__ chain contains a cycle
    must not loop forever inside safe_traceback."""

    test_logger = logging.getLogger("test_pii_async_cycle")
    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    test_logger.addHandler(_Capture())
    test_logger.setLevel(logging.ERROR)

    e1 = RuntimeError("first")
    e2 = RuntimeError("second")
    e1.__context__ = e2
    e2.__context__ = e1  # cycle: e1 -> e2 -> e1

    def handler(request):
        return httpx.Response(200, json={"redactedText": "ok"})

    _install_mock_async_client(handler)
    await log_error_strip_pii(test_logger, "X failed", exc=e1)

    msg = records[-1].getMessage()
    # Both types appear (chain is walked) but the function returned —
    # didn't loop forever. The cycle detection in safe_traceback breaks
    # on the revisit.
    assert "RuntimeError" in msg
    # Chain rendered as `RuntimeError -> RuntimeError` (two visits, then
    # cycle detected on third). Should NOT contain the messages.
    assert "first" not in msg
    assert "second" not in msg


async def test_log_error_strip_pii_propagates_extra_to_log_record():
    """The extra kwarg should attach structured fields to the LogRecord
    so Splunk indexes them directly without relying on key=value
    auto-extraction from the message body."""

    def handler(request):
        return httpx.Response(200, json={"redactedText": "ok"})

    _install_mock_async_client(handler)
    test_logger = logging.getLogger("test_pii_async_extra")
    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    test_logger.addHandler(_Capture())
    test_logger.setLevel(logging.ERROR)

    try:
        _raise_with_user_msg("anything")
    except ValueError as e:
        await log_error_strip_pii(
            test_logger,
            "X failed",
            exc=e,
            extra={"client_id": "tinypsa", "user_id": "u_456"},
        )

    rec = records[-1]
    assert rec.client_id == "tinypsa"
    assert rec.user_id == "u_456"


async def test_log_error_strip_pii_injects_error_type_and_masked_error():
    """error_type and error (masked) are auto-injected as structured
    fields so Splunk dashboards that key on `error=...` can rely on
    that field — symmetric with the ECMS helper."""

    def handler(request):
        return httpx.Response(200, json={"redactedText": "ssn ***-**-6789"})

    _install_mock_async_client(handler)
    test_logger = logging.getLogger("test_pii_async_error_field")
    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    test_logger.addHandler(_Capture())
    test_logger.setLevel(logging.ERROR)

    runtime_pii = "user " + "ssn=" + "1" + "23-45-6789"
    try:
        _raise_with_user_msg(runtime_pii)
    except ValueError as e:
        await log_error_strip_pii(
            test_logger,
            "X failed",
            exc=e,
            extra={"client_id": "tinypsa"},
        )

    rec = records[-1]
    # Auto-injected.
    assert rec.error_type == "ValueError"
    assert rec.error == "ssn ***-**-6789"
    # Caller-supplied still present.
    assert rec.client_id == "tinypsa"
    # Unredacted runtime value never reaches the LogRecord.
    assert "123-45-6789" not in rec.error


async def test_log_error_strip_pii_caller_extra_overrides_auto_injected():
    """Caller-supplied error/error_type take precedence over auto-injection."""

    def handler(request):
        return httpx.Response(200, json={"redactedText": "ok"})

    _install_mock_async_client(handler)
    test_logger = logging.getLogger("test_pii_async_override")
    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    test_logger.addHandler(_Capture())
    test_logger.setLevel(logging.ERROR)

    try:
        _raise_with_user_msg("anything")
    except ValueError as e:
        await log_error_strip_pii(
            test_logger,
            "X failed",
            exc=e,
            extra={"error": "caller-supplied", "error_type": "CallerType"},
        )

    rec = records[-1]
    assert rec.error == "caller-supplied"
    assert rec.error_type == "CallerType"


# get_ispy_pii_auth_headers


@pytest.mark.asyncio(loop_scope="function")
async def test_auth_headers_empty_without_creds(monkeypatch):
    """No env vars => no header."""
    from mirix.pii import get_ispy_pii_auth_headers

    monkeypatch.delenv("MIRIX_ISPY_PII_APPID", raising=False)
    monkeypatch.delenv("MIRIX_ISPY_PII_APP_SECRET", raising=False)
    assert get_ispy_pii_auth_headers() == {}


@pytest.mark.asyncio(loop_scope="function")
async def test_auth_headers_empty_with_partial_creds(monkeypatch):
    """Only one of the two env vars set => no header."""
    from mirix.pii import get_ispy_pii_auth_headers

    monkeypatch.setenv("MIRIX_ISPY_PII_APPID", "Intuit.test")
    monkeypatch.delenv("MIRIX_ISPY_PII_APP_SECRET", raising=False)
    assert get_ispy_pii_auth_headers() == {}

    monkeypatch.delenv("MIRIX_ISPY_PII_APPID", raising=False)
    monkeypatch.setenv("MIRIX_ISPY_PII_APP_SECRET", "secret")
    assert get_ispy_pii_auth_headers() == {}


@pytest.mark.asyncio(loop_scope="function")
async def test_auth_headers_with_creds(monkeypatch):
    """Both env vars => Intuit_IAM_Authentication header."""
    from mirix.pii import get_ispy_pii_auth_headers

    monkeypatch.setenv(
        "MIRIX_ISPY_PII_APPID",
        "Intuit.expertise.help.contextandmemoryservice",
    )
    monkeypatch.setenv("MIRIX_ISPY_PII_APP_SECRET", "s3cr3t")
    headers = get_ispy_pii_auth_headers()
    assert headers["Authorization"] == (
        "Intuit_IAM_Authentication intuit_appid=Intuit.expertise.help.contextandmemoryservice,intuit_app_secret=s3cr3t"
    )
    assert headers["intuit_offeringid"] == ("Intuit.expertise.help.contextandmemoryservice")


@pytest.mark.asyncio(loop_scope="function")
async def test_async_client_sends_auth_header(monkeypatch):
    """Boundary: headers actually reach the wire."""
    import mirix.pii as _pii

    monkeypatch.setenv("MIRIX_ISPY_PII_APPID", "Intuit.test")
    monkeypatch.setenv("MIRIX_ISPY_PII_APP_SECRET", "shh")

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers.get("authorization", "")
        captured["intuit_offeringid"] = request.headers.get("intuit_offeringid", "")
        return httpx.Response(200, json={"redactedText": "ok"})

    transport = httpx.MockTransport(handler)
    _pii._async_client = httpx.AsyncClient(
        transport=transport,
        timeout=0.2,
        follow_redirects=True,
        headers=_pii.get_ispy_pii_auth_headers(),
    )
    masked = await _pii._mask_async("hello world")
    assert masked == "ok"
    assert captured["authorization"].startswith("Intuit_IAM_Authentication ")
    assert "intuit_appid=Intuit.test" in captured["authorization"]
    assert "intuit_app_secret=shh" in captured["authorization"]
    assert captured["intuit_offeringid"] == "Intuit.test"


# mask_structure — recursive cooperative async masker


async def test_mask_structure_masks_leaf_string():
    """A bare leaf string is forwarded to ispy-pii and the redacted
    value is returned."""

    def handler(request):
        return httpx.Response(200, json={"redactedText": "ssn ***-**-6789"})

    _install_mock_async_client(handler)
    assert await mask_structure("ssn 123-45-6789") == "ssn ***-**-6789"


async def test_mask_structure_walks_nested_dict_list_tuple():
    """Every leaf string inside a nested dict/list/tuple is masked while
    the container shapes (dict keys, list, tuple) are preserved."""
    seen: list = []

    def handler(request):
        body = request.read().decode()
        seen.append(body)
        return httpx.Response(200, json={"redactedText": "X"})

    _install_mock_async_client(handler)
    result = await mask_structure(
        {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ],
            "pair": ("a", "b"),
        }
    )
    assert result == {
        "messages": [
            {"role": "X", "content": "X"},
            {"role": "X", "content": "X"},
        ],
        "pair": ("X", "X"),
    }
    # Tuple stays a tuple (not coerced to a list).
    assert isinstance(result["pair"], tuple)
    # Six leaf strings → six ispy-pii calls.
    assert len(seen) == 6


async def test_mask_structure_passes_through_non_string_scalars():
    """int / float / bool / None pass through unchanged and never hit
    the network."""

    def handler(request):
        return httpx.Response(200, json={"redactedText": "should-not-be-called"})

    _install_mock_async_client(handler)
    result = await mask_structure({"n": 42, "f": 3.14, "b": True, "z": None})
    assert result == {"n": 42, "f": 3.14, "b": True, "z": None}


async def test_mask_structure_fails_closed_per_leaf_to_placeholder():
    """A leaf whose ispy-pii call errors (5xx) degrades to the
    placeholder; mask_structure never raises."""

    def handler(request):
        return httpx.Response(500, text="boom")

    _install_mock_async_client(handler)
    result = await mask_structure({"messages": ["leaks here"]})
    assert result == {"messages": [REDACTED_PLACEHOLDER]}


async def test_mask_structure_passthrough_when_disabled(monkeypatch):
    """Kill switch off → full passthrough, no network call."""
    monkeypatch.setenv("MIRIX_ISPY_PII_ENABLED", "false")

    def handler(request):
        return httpx.Response(200, json={"redactedText": "should-not-be-called"})

    _install_mock_async_client(handler)
    payload = {"messages": [{"role": "user", "content": "francis@example.com"}]}
    assert await mask_structure(payload) == payload


async def test_mask_structure_masks_sibling_leaves_concurrently():
    """K sibling leaves issue ~K concurrent calls. The handler blocks
    until all K requests have arrived, so a serial implementation would
    deadlock/timeout; only a concurrent gather lets all K arrive at once."""
    import asyncio

    n_leaves = 5
    arrived = asyncio.Event()
    in_flight = 0
    max_in_flight = 0

    async def handler(request):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        if in_flight >= n_leaves:
            arrived.set()
        # Wait until every sibling leaf is concurrently in flight.
        await asyncio.wait_for(arrived.wait(), timeout=2.0)
        in_flight -= 1
        return httpx.Response(200, json={"redactedText": "X"})

    _install_mock_async_client(handler)
    leaves = [f"leaf-{i}" for i in range(n_leaves)]
    result = await mask_structure(leaves)
    assert result == ["X"] * n_leaves
    # All leaves were in flight simultaneously → truly concurrent.
    assert max_in_flight == n_leaves
