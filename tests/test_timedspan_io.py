"""Input/output behavior of ``timedspan`` (R3 mechanism).

Before this change, ``timedspan`` accepted only ``name`` + ``metadata`` — no
way to attach span input or output — which is why 20+ save-path spans rendered
``Input: undefined / Output: undefined`` in LangFuse. The mechanism fix:

- ``metadata`` mirrors to the span ``input`` by default (the metadata at these
  call sites already IS "what the step operated on"); ``input=None`` suppresses
  the mirror; an explicit ``input=`` overrides it.
- The body sets output via the reserved ``rec["span_output"]`` key
  (context-manager form) or ``record_output(**fields)`` (decorator form).
- On exception the span gets ``level=ERROR`` with the exception TYPE NAME only
  (``str(e)`` can echo user content — PII posture).
- Every new span interaction extends the existing never-raise contract: the
  wrapped work's outcome always wins over instrumentation failures.
"""

from unittest.mock import MagicMock, patch

import pytest

from mirix.observability import context as obs_context
from mirix.observability.timed import record_output, timedspan


@pytest.fixture(autouse=True)
def _trace_context():
    obs_context.clear_trace_context()
    obs_context.set_trace_context(trace_id="trace-1", observation_id="obs-1")
    yield
    obs_context.clear_trace_context()


def _make_langfuse():
    span = MagicMock()
    span.id = "span-1"
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=span)
    cm.__exit__ = MagicMock(return_value=False)
    langfuse = MagicMock()
    langfuse.start_as_current_observation.return_value = cm
    return langfuse, span


def _patched(langfuse):
    return patch("mirix.observability.timed.get_langfuse_client", return_value=langfuse)


# --------------------------------------------------------------------------- #
# Input mirroring
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_metadata_mirrors_to_input_by_default():
    langfuse, _span = _make_langfuse()
    with _patched(langfuse):
        async with timedspan("Load Agent State", metadata={"agent_id": "a-1"}):
            pass

    kwargs = langfuse.start_as_current_observation.call_args.kwargs
    assert kwargs["input"] == {"agent_id": "a-1"}


@pytest.mark.asyncio
async def test_input_none_suppresses_the_mirror():
    langfuse, _span = _make_langfuse()
    with _patched(langfuse):
        async with timedspan("Load Agent", metadata={"agent_id": "a-1"}, input=None):
            pass

    kwargs = langfuse.start_as_current_observation.call_args.kwargs
    assert "input" not in kwargs


@pytest.mark.asyncio
async def test_explicit_input_overrides_the_mirror():
    langfuse, _span = _make_langfuse()
    with _patched(langfuse):
        async with timedspan("Inner Step", metadata={"step_count": 0}, input={"custom": True}):
            pass

    kwargs = langfuse.start_as_current_observation.call_args.kwargs
    assert kwargs["input"] == {"custom": True}


@pytest.mark.asyncio
async def test_no_metadata_means_no_input():
    """With no metadata there is nothing to mirror — don't attach input={}."""
    langfuse, _span = _make_langfuse()
    with _patched(langfuse):
        async with timedspan("Bare Span"):
            pass

    kwargs = langfuse.start_as_current_observation.call_args.kwargs
    assert "input" not in kwargs


@pytest.mark.asyncio
async def test_mirror_does_not_include_the_stamped_tid():
    """The tid is stamped into span METADATA for FST capture; the input mirror
    reflects the caller's metadata only."""
    langfuse, _span = _make_langfuse()
    with (
        _patched(langfuse),
        patch("mirix.observability.context.get_tid", return_value="tid-1"),
    ):
        async with timedspan("Retrieve core", metadata={"backend": "ipsr"}):
            pass

    kwargs = langfuse.start_as_current_observation.call_args.kwargs
    assert kwargs["metadata"] == {"backend": "ipsr", "tid": "tid-1"}
    assert kwargs["input"] == {"backend": "ipsr"}


# --------------------------------------------------------------------------- #
# Output channel — context-manager form
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_span_output_rec_key_delivered_as_span_output():
    langfuse, span = _make_langfuse()
    with _patched(langfuse):
        async with timedspan("Load Retained History", metadata={"limit": 5}) as rec:
            rec["span_output"] = {"loaded_count": 3}

    span.update.assert_any_call(output={"loaded_count": 3})


@pytest.mark.asyncio
async def test_span_output_popped_before_timing_line():
    """The reserved key is consumed on exit so a custom ``line`` callback (which
    receives ``rec``) never sees it."""
    langfuse, _span = _make_langfuse()
    seen = {}

    def _line(ms, rec):
        seen["rec_keys"] = set(rec.keys())
        return "timing"

    with _patched(langfuse):
        async with timedspan("Persist Memory Source", line=_line) as rec:
            rec["span_output"] = {"persisted": True}
            rec["other"] = 1

    assert "span_output" not in seen["rec_keys"]
    assert "other" in seen["rec_keys"]


@pytest.mark.asyncio
async def test_no_output_update_when_body_sets_none():
    langfuse, span = _make_langfuse()
    with _patched(langfuse):
        async with timedspan("Agent Step", metadata={"agent_type": "meta"}):
            pass

    for call in span.update.call_args_list:
        assert "output" not in call.kwargs


# --------------------------------------------------------------------------- #
# Output channel — decorator form (record_output)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_record_output_decorator_form_parity():
    langfuse, span = _make_langfuse()

    @timedspan("Decorated Op", metadata={"k": "v"})
    async def _op():
        record_output(tools_loaded=7)
        record_output(chunks=2)

    with _patched(langfuse):
        await _op()

    span.update.assert_any_call(output={"tools_loaded": 7, "chunks": 2})


def test_record_output_outside_active_record_is_noop():
    record_output(x=1)  # must not raise


# --------------------------------------------------------------------------- #
# Exception path
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_exception_sets_error_level_with_type_name_only():
    langfuse, span = _make_langfuse()
    secret = "SENSITIVE-USER-CONTENT"
    with _patched(langfuse):
        with pytest.raises(ValueError):
            async with timedspan("Persist Memory Source", metadata={"m": 1}):
                raise ValueError(secret)

    error_calls = [c for c in span.update.call_args_list if c.kwargs.get("level") == "ERROR"]
    assert error_calls, "expected a level=ERROR span update on exception"
    assert error_calls[0].kwargs.get("status_message") == "ValueError"
    # PII posture: str(e) never reaches the span.
    for call in span.update.call_args_list:
        assert secret not in repr(call)


# --------------------------------------------------------------------------- #
# No-op / never-raise guarantees
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_noop_when_langfuse_disabled_output_key_is_harmless():
    with patch("mirix.observability.timed.get_langfuse_client", return_value=None):
        async with timedspan("X", metadata={"a": 1}) as rec:
            rec["span_output"] = {"b": 2}
            ran = True
    assert ran


@pytest.mark.asyncio
async def test_noop_when_no_trace_context():
    obs_context.clear_trace_context()
    langfuse, _span = _make_langfuse()
    with _patched(langfuse):
        async with timedspan("X", metadata={"a": 1}) as rec:
            rec["span_output"] = {"b": 2}
    langfuse.start_as_current_observation.assert_not_called()


@pytest.mark.asyncio
async def test_noop_when_span_creation_fails():
    langfuse, _span = _make_langfuse()
    langfuse.start_as_current_observation.side_effect = RuntimeError("boom")
    with _patched(langfuse):
        async with timedspan("X", metadata={"a": 1}) as rec:
            rec["span_output"] = {"b": 2}
            ran = True
    assert ran


@pytest.mark.asyncio
async def test_output_update_failure_never_raises():
    langfuse, span = _make_langfuse()
    span.update.side_effect = RuntimeError("update boom")
    with _patched(langfuse):
        async with timedspan("X", metadata={"a": 1}) as rec:
            rec["span_output"] = {"b": 2}


@pytest.mark.asyncio
async def test_output_update_failure_does_not_mask_body_exception():
    """The never-raise contract of the exit path extends to the new
    span.update: a faulty update must not REPLACE the body's real exception."""
    langfuse, span = _make_langfuse()
    span.update.side_effect = RuntimeError("update boom")
    with _patched(langfuse):
        with pytest.raises(KeyError):
            async with timedspan("X", metadata={"a": 1}) as rec:
                rec["span_output"] = {"b": 2}
                raise KeyError("real failure")
