"""Tests for the ``timed`` / ``timedspan`` timing+tracing construct."""

import logging

import pytest

from mirix.observability.timed import record_timing, timed, timedspan

pytestmark = pytest.mark.asyncio


async def _noop_block():
    return None


def _records(caplog):
    return [r for r in caplog.records if "TIMING" in r.getMessage()]


async def test_timed_context_manager_logs_default_line_at_debug(caplog):
    log = logging.getLogger("test.timed")
    with caplog.at_level(logging.DEBUG, logger="test.timed"):
        async with timed("op_a", logger=log):
            await _noop_block()
    recs = _records(caplog)
    assert len(recs) == 1
    assert recs[0].levelno == logging.DEBUG
    assert recs[0].getMessage().startswith("[op_a TIMING] execute_ms=")


async def test_timed_warns_when_slow_ms_crossed(caplog):
    log = logging.getLogger("test.timed")
    with caplog.at_level(logging.DEBUG, logger="test.timed"):
        async with timed("op_b", logger=log, slow_ms=0.0):
            await _noop_block()
    recs = _records(caplog)
    assert len(recs) == 1
    assert recs[0].levelno == logging.WARNING
    assert recs[0].getMessage().endswith(" SLOW")


async def test_timed_no_slow_ms_never_warns(caplog):
    log = logging.getLogger("test.timed")
    with caplog.at_level(logging.DEBUG, logger="test.timed"):
        async with timed("op_c", logger=log):
            await _noop_block()
    assert all(r.levelno == logging.DEBUG for r in _records(caplog))


async def test_custom_line_receives_ms_and_rec(caplog):
    log = logging.getLogger("test.timed")
    with caplog.at_level(logging.DEBUG, logger="test.timed"):
        async with timed(
            "op_d", logger=log, line=lambda ms, rec: f"[CUSTOM] hits={rec['hits']} execute_ms={ms:.1f}"
        ) as rec:
            rec["hits"] = 7
    recs = [r for r in caplog.records if "CUSTOM" in r.getMessage()]
    assert any("hits=7" in r.getMessage() and "execute_ms=" in r.getMessage() for r in recs)


async def test_raising_body_still_logs(caplog):
    log = logging.getLogger("test.timed")
    with caplog.at_level(logging.DEBUG, logger="test.timed"):
        with pytest.raises(ValueError):
            async with timed("op_e", logger=log):
                raise ValueError("boom")
    assert len(_records(caplog)) == 1


async def test_faulty_line_never_masks_propagating_exception(caplog):
    """A ``line`` callback that blows up while the body is unwinding an
    exception must NOT replace the in-flight exception.

    Regression: the IPS Search timing line does ``rec['hits']`` but only seeds
    that key AFTER the fallible ``_post_json`` call. When ``_post_json`` raised,
    the ``finally`` evaluated the line on an empty ``rec`` and the resulting
    ``KeyError: 'hits'`` clobbered the real (transient) error all the way up the
    agent stack. Instrumentation must never mask real failures.
    """
    log = logging.getLogger("test.timed")
    with caplog.at_level(logging.DEBUG, logger="test.timed"):
        with pytest.raises(ValueError, match="real failure"):
            async with timed(
                "op_mask",
                logger=log,
                line=lambda ms, rec: f"[X] hits={rec['hits']}",
            ) as rec:
                # body raises before ever seeding rec["hits"]
                raise ValueError("real failure")
                rec["hits"] = 0  # noqa: unreachable


async def test_faulty_line_on_success_does_not_break_block(caplog):
    """Even on the happy path, a ``line`` that raises must be swallowed so the
    timed block returns normally (the work already succeeded)."""
    log = logging.getLogger("test.timed")
    with caplog.at_level(logging.DEBUG, logger="test.timed"):
        async with timed(
            "op_mask_ok",
            logger=log,
            line=lambda ms, rec: f"[X] missing={rec['nope']}",
        ):
            await _noop_block()
    # No exception escaped; block completed.


async def test_decorator_form_logs_and_record_timing_reaches_line(caplog):
    log = logging.getLogger("test.timed")

    @timed("decorated", logger=log, line=lambda ms, rec: f"[DEC] n={rec.get('n')} execute_ms={ms:.1f}")
    async def do_work():
        record_timing(n=3)
        return "ok"

    with caplog.at_level(logging.DEBUG, logger="test.timed"):
        result = await do_work()
    assert result == "ok"
    assert any("n=3" in r.getMessage() for r in caplog.records)


async def test_timedspan_logs_line_even_without_tracing(caplog):
    log = logging.getLogger("test.timed")
    with caplog.at_level(logging.DEBUG, logger="test.timed"):
        async with timedspan("SpanOp", {"k": "v"}, logger=log):
            await _noop_block()
    recs = _records(caplog)
    assert len(recs) == 1
    assert recs[0].getMessage().startswith("[SpanOp TIMING] execute_ms=")
