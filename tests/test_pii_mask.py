"""Tests for mirix.observability.pii_mask.

real PII masking now happens UPSTREAM (mirix.pii.mask_structure
at the LLM/embedding generation sites). The Langfuse ``mask=`` callback is
reduced to a cheap **synchronous backstop**. It must:
- Do NO network I/O (no httpx.Client, no ispy-pii POST) — the SDK invokes it
  synchronously on the event-loop thread, so a blocking call there starves
  the loop (the exact bug this story fixes).
- Still walk dicts/lists/tuples/str so any attribute NOT pre-masked has a
  safety net.
- Locally regex-scrub the obvious high-risk tokens (email / SSN / phone) with
  zero network.
- Honor the MIRIX_LANGFUSE_MASK_ENABLED kill switch (passthrough).
- Preserve the set_langfuse_mask / get_langfuse_mask / ispy_pii_mask seam so
  downstream consumers can register their own synchronous callable.
"""

import pytest

from mirix.observability.pii_mask import (
    build_langfuse_mask,
    get_langfuse_mask,
    ispy_pii_mask,
    set_langfuse_mask,
)


@pytest.fixture(autouse=True)
def _enable_mask(monkeypatch):
    monkeypatch.setenv("MIRIX_LANGFUSE_MASK_ENABLED", "true")
    yield


@pytest.fixture(autouse=True)
def _reset_active_mask():
    """Module-level _active_mask must not leak across tests."""
    yield
    set_langfuse_mask(None)


# --- No network: the backstop holds no client and issues no requests ---


def test_module_holds_no_httpx_client():
    """The module must not even import/construct httpx for the mask path."""
    import mirix.observability.pii_mask as pm

    assert not hasattr(pm, "httpx")


def test_mask_does_no_network_for_any_input():
    """Patching httpx.Client to explode proves the backstop never touches
    the network for strings, dicts, lists, or tuples."""
    import httpx

    def _boom(*a, **k):
        raise AssertionError("mask must not open an httpx.Client")

    mask = build_langfuse_mask()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(httpx, "Client", _boom)
        assert mask(data="plain text") == "plain text"
        assert mask(data={"k": "v"}) == {"k": "v"}
        assert mask(data=["a", "b"]) == ["a", "b"]
        assert mask(data=("a", "b")) == ("a", "b")


# --- Structure walking ---


def test_mask_walks_nested_dict_list_tuple():
    mask = build_langfuse_mask()
    result = mask(
        data={
            "messages": [{"role": "user", "content": "hello"}],
            "pair": ("x", "y"),
        }
    )
    assert result == {
        "messages": [{"role": "user", "content": "hello"}],
        "pair": ("x", "y"),
    }
    assert isinstance(result["pair"], tuple)


def test_mask_passes_through_non_string_scalars():
    mask = build_langfuse_mask()
    assert mask(data=42) == 42
    assert mask(data=3.14) == 3.14
    assert mask(data=True) is True
    assert mask(data=None) is None


def test_mask_handles_empty_string():
    mask = build_langfuse_mask()
    assert mask(data="") == ""


# --- Local regex backstop scrubs email / SSN / phone (zero network) ---


def test_backstop_scrubs_email():
    mask = build_langfuse_mask()
    out = mask(data="contact me at francis@example.com please")
    assert "francis@example.com" not in out
    assert "contact me at" in out


def test_backstop_scrubs_ssn():
    mask = build_langfuse_mask()
    out = mask(data="my ssn is 123-45-6789 ok")
    assert "123-45-6789" not in out


def test_backstop_scrubs_phone():
    mask = build_langfuse_mask()
    out = mask(data="call 415-555-0182 now")
    assert "415-555-0182" not in out


def test_backstop_scrubs_pii_nested_in_structure():
    mask = build_langfuse_mask()
    out = mask(data={"messages": ["email bob@acme.com", {"phone": "415-555-0182"}]})
    flat = repr(out)
    assert "bob@acme.com" not in flat
    assert "415-555-0182" not in flat


def test_backstop_leaves_non_pii_text_untouched():
    mask = build_langfuse_mask()
    assert mask(data="the quick brown fox") == "the quick brown fox"


# --- Kill switch ---


def test_mask_passthrough_when_disabled(monkeypatch):
    monkeypatch.setenv("MIRIX_LANGFUSE_MASK_ENABLED", "false")
    mask = build_langfuse_mask()
    # Even raw PII passes through verbatim when disabled.
    assert mask(data="user francis@example.com") == "user francis@example.com"
    assert mask(data={"k": "v"}) == {"k": "v"}
    assert mask(data=["a", "b"]) == ["a", "b"]


# --- Seam: set/get_langfuse_mask + ispy_pii_mask default ---


def test_get_langfuse_mask_falls_back_to_default_when_unset():
    set_langfuse_mask(None)
    assert get_langfuse_mask() is ispy_pii_mask


def test_set_langfuse_mask_overrides_default():
    sentinel_calls: list = []

    def sentinel(data, **kwargs):
        sentinel_calls.append(data)
        return data

    set_langfuse_mask(sentinel)
    assert get_langfuse_mask() is sentinel
    get_langfuse_mask()(data="hello")
    assert sentinel_calls == ["hello"]


def test_set_langfuse_mask_none_clears_registration():
    set_langfuse_mask(lambda data, **kwargs: data)
    set_langfuse_mask(None)
    assert get_langfuse_mask() is ispy_pii_mask


def test_ispy_pii_mask_default_scrubs_email_without_network():
    """The default callable consulted by get_langfuse_mask is the cheap
    synchronous backstop and scrubs email locally."""
    out = ispy_pii_mask(data="reach me at francis@example.com")
    assert "francis@example.com" not in out
