"""The gateway ledger summary of ``cell_records.py`` (H9 cost inputs): cut-off calls whose
cost is unknown, cache writes at their list price, the reasoning tokens and slow calls per
provider and a call's completion time."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from evals.skill_bench import protocol
from evals.skill_bench.cell_records import (
    REASONING_EDGES,
    SLOW_CALL_DEADLINE_FRACTION,
    call_completed_at,
    list_price_usd,
    summarize_ledger,
    unknown_cost_attempts,
)

LUNA = protocol.MODELS["luna"]
LUNA56 = protocol.MODELS["luna56"]


def _write(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _call(call_id: str, role: str, phase: str, **kw: Any) -> dict[str, Any]:
    return {
        "event": "call",
        "call_id": call_id,
        "role": role,
        "run_phase": phase,
        "status": "ok",
        "cost_total": 0.01,
        "billed_attempts": 1,
        "model": LUNA,
        "provider": "P1",
        **kw,
    }


def _deadline(call_id: str, role: str, phase: str, **kw: Any) -> dict[str, Any]:
    fields = {
        "status": "deadline_exceeded",
        "provider": None,
        "cost_total": None,
        "billed_attempts": 0,
    }
    return _call(call_id, role, phase, **{**fields, **kw})


# --- deadline calls: billed, cost unknown ------------------------------------------------------


def test_unknown_cost_attempts_from_the_fields_or_the_attempts():
    """New lines carry the count of every attempt cut off after its request was sent (the
    deadline, or a dropped connection); lines before that carry the deadline
    ones; older lines are counted from their attempts' outcome, which the gateway has
    always written (an upper bound: such a line cannot tell a pool or write timeout,
    never sent, apart); a line without either counts none."""
    assert unknown_cost_attempts({"unknown_cost_attempts": 3}) == 3
    assert unknown_cost_attempts({"unbilled_deadline_attempts": 2}) == 2
    # the newest field wins, and a field wins over the attempts
    assert (
        unknown_cost_attempts(
            {"unknown_cost_attempts": 2, "unbilled_deadline_attempts": 1}
        )
        == 2
    )
    assert (
        unknown_cost_attempts(
            {"unbilled_deadline_attempts": 0, "attempts": [{"outcome": "deadline"}]}
        )
        == 0
    )
    legacy = {"attempts": [{"outcome": "infra"}, {"outcome": "deadline"}]}
    assert unknown_cost_attempts(legacy) == 1
    assert unknown_cost_attempts({"status": "ok"}) == 0
    assert unknown_cost_attempts({"attempts": "garbage"}) == 0
    assert unknown_cost_attempts({"unbilled_deadline_attempts": True}) == 0


def test_the_summary_counts_calls_of_unknown_cost_per_phase_role_and_item(tmp_path):
    path = _write(
        tmp_path / "gateway.jsonl",
        [
            _call("a", "actor", "learning", item="l1"),
            _deadline(
                "b",
                "learner",
                "learning",
                attempts=[{"outcome": "deadline"}],  # an older line: no count field
            ),
            _call("c", "actor", "test", item="u1", unbilled_deadline_attempts=0),
            _deadline(
                "d",
                "actor",
                "test",
                item="u1",
                attempts=[{"outcome": "deadline", "billed": "unknown"}],
                unbilled_deadline_attempts=1,
            ),
            _deadline(  # a billed retry, then the deadline: cost known in part
                "e",
                "actor",
                "test",
                item="u2",
                cost_total=0.004,
                billed_attempts=1,
                attempts=[
                    {"outcome": "choice_error", "cost": 0.004},
                    {"outcome": "deadline", "billed": "unknown"},
                ],
                unbilled_deadline_attempts=1,
            ),
            _deadline(  # timed out while queued: no attempt ran, nothing billed
                "f",
                "actor",
                "test",
                item="u3",
                attempts=[],
                unbilled_deadline_attempts=0,
            ),
        ],
    )
    summary = summarize_ledger(path)
    by = summary["by_phase_role"]
    assert by["learning/actor"]["unknown_cost_calls"] == 0
    assert by["learning/learner"]["unknown_cost_calls"] == 1
    assert by["learning/learner"]["unknown_cost_attempts"] == 1
    assert by["test/actor"]["unknown_cost_calls"] == 2
    assert by["test/actor"]["cost_total"] == pytest.approx(
        0.014
    )  # c + e's billed retry
    assert summary["totals"]["unknown_cost_calls"] == 3
    assert summary["totals"]["billed_attempts"] == 3
    assert summary["items"]["u2"] == {"test": 0.004}
    assert summary["items_unknown_cost"] == {"u1": {"test": 1}, "u2": {"test": 1}}


def test_a_ledger_without_deadlines_reads_as_before(tmp_path):
    summary = summarize_ledger(
        _write(tmp_path / "gateway.jsonl", [_call("a", "actor", "test", item="u1")])
    )
    assert summary["totals"]["unknown_cost_calls"] == 0
    assert summary["items_unknown_cost"] == {}
    assert summary["items"] == {"u1": {"test": 0.01}}


# --- provider routing the calls carried ----------------------------------------------------


def test_the_routing_each_call_carried_per_model(tmp_path):
    """``routing`` counts per model the ``wire.provider`` each call went out with
    (``none`` without one); ``routing_unrecorded`` the lines of a sent request (it has a
    hash) that do not record ``wire``; a call refused before the rewrite, which sent
    nothing, and a ``call_started`` line are in neither."""
    policy = {"ignore": ["wafer", "open-inference"]}
    rows = [
        _call(
            "a",
            "actor",
            "test",
            model=LUNA56,
            request_sha256="1",
            wire={"model": LUNA56, "provider": policy},
        ),
        _call(
            "b",
            "actor",
            "test",
            model=LUNA56,
            request_sha256="2",
            wire={"model": LUNA56, "provider": policy},
        ),
        _call(
            "c",
            "actor",
            "test",
            model=LUNA56,
            request_sha256="3",
            wire={"model": LUNA56},
        ),
        _call("d", "actor", "test", request_sha256="4", wire={"model": LUNA}),
        _call("e", "actor", "test", request_sha256="5"),
        _call("f", "actor", "test", status="bad_request", cost_total=0.0),
        {
            "event": "call_started",
            "call_id": "g",
            "model": LUNA56,
            "request_sha256": "6",
        },
    ]
    summary = summarize_ledger(_write(tmp_path / "gateway.jsonl", rows))
    assert summary["routing"] == {
        LUNA: {"none": 1},
        LUNA56: {'{"ignore": ["wafer", "open-inference"]}': 2, "none": 1},
    }
    assert summary["routing_unrecorded"] == {LUNA: 1}


# --- list price: cache writes ------------------------------------------------------------------


def test_cache_writes_are_priced_at_input_cache_write():
    """luna's prompt tokens written to the cache bill at its $0.125/M list price, not the
    $0.10/M prompt price; they are part of the prompt tokens."""
    listed, over = list_price_usd(LUNA, 1100, 0, 100, 1000)
    assert listed == pytest.approx(100 * 0.1e-6 + 1000 * 0.125e-6 + 100 * 0.5e-6)
    assert listed == pytest.approx(0.000185) and not over
    listed, _ = list_price_usd(LUNA, 300_000, 0, 0, 100_000)  # the long-input tier
    assert listed == pytest.approx(200_000 * 0.2e-6 + 100_000 * 0.25e-6)
    assert list_price_usd(LUNA, 10, 0, 0) == list_price_usd(LUNA, 10, 0, 0, 0)


def test_the_pricing_entry_reads_cache_writes_from_the_usage(tmp_path):
    row = _call(
        "a",
        "actor",
        "test",
        provider="OpenAI",
        cost=0.000185,
        input_tokens_actual=1100,
        cached_tokens=0,
        output_tokens=100,
        usage={
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 1000}
        },
        attempts=[{"outcome": "ok", "latency_s": 2.4}],
    )
    entry = summarize_ledger(_write(tmp_path / "gateway.jsonl", [row]))["pricing"][
        LUNA
    ]["OpenAI"]
    assert entry["ratio_min"] == entry["ratio_max"] == 1.0
    assert entry["cost_usd"] == pytest.approx(entry["list_cost_usd"])


# --- reasoning tokens and slow calls -----------------------------------------------------------


def test_reasoning_tokens_and_slow_calls_per_provider(tmp_path):
    slow = protocol.CALL_DEADLINE_S * SLOW_CALL_DEADLINE_FRACTION
    rows = [
        _call(
            "a",
            "actor",
            "test",
            model=LUNA56,
            provider="Wafer",
            reasoning_tokens=25_000,
            attempts=[{"outcome": "ok", "latency_s": slow + 100.0}],
        ),
        _call(
            "b",
            "actor",
            "test",
            model=LUNA56,
            provider="Wafer",
            reasoning_tokens=0,
            # the call took long, its answering attempt did not: not the provider's time
            latency_s=slow + 50,
            attempts=[
                {"outcome": "infra", "latency_s": slow},
                {"outcome": "ok", "latency_s": 12.0},
            ],
        ),
        _call(
            "c",
            "actor",
            "test",
            model=LUNA56,
            provider="Wafer",
            attempts=[
                {"outcome": "ok", "latency_s": slow}
            ],  # exactly half: not past it
        ),
    ]
    entry = summarize_ledger(_write(tmp_path / "gateway.jsonl", rows))["pricing"][
        LUNA56
    ]["Wafer"]
    assert entry["calls"] == 3 and entry["slow_calls"] == 1
    assert entry["reasoning_calls"] == 2 and entry["reasoning_tokens"] == 25_000
    assert entry["reasoning_tokens_max"] == 25_000
    buckets = entry["reasoning_buckets"]
    assert len(buckets) == len(REASONING_EDGES) + 1 and sum(buckets) == 2
    assert buckets[0] == 1  # 0 tokens
    assert buckets[REASONING_EDGES.index(50_000)] == 1  # 20,000 < 25,000 <= 50,000


# --- completion time ---------------------------------------------------------------------------


def test_a_call_completed_at_its_completed_at_or_ts_plus_latency():
    at = datetime(2026, 9, 29, 5, 49, tzinfo=timezone.utc)
    assert (
        call_completed_at(
            {"ts": "2026-09-29T05:39:00+00:00", "completed_at": at.isoformat()}
        )
        == at
    )
    # an older line: arrival plus latency (a 600 s deadline call ended 600 s later)
    assert (
        call_completed_at({"ts": "2026-09-29T05:39:00+00:00", "latency_s": 600.0}) == at
    )
    assert call_completed_at({"ts": "2026-09-29T05:49:00"}) == at  # no zone: UTC
    assert call_completed_at({"ts": "garbage", "latency_s": 1}) is None
    assert call_completed_at({}) is None


# --- learning checkpoints and resumes -----------------------------------------------------


def test_the_shared_status_names_and_exit_codes():
    from evals.skill_bench import cell_records as cr

    assert cr.STATUS_LEARNING_INTERRUPTED == "learning_interrupted"
    assert cr.EXIT_CODES["learning_interrupted"] == 8
    assert list(cr.EXIT_CODES.values()).count(8) == 1  # its own code
    assert cr.AUTO_RESUMABLE_STATUSES == {
        "learning_interrupted",
        "test_incomplete",
        "infra_unavailable",
        "interrupted",
        "crashed",
    }
    assert cr.AUTO_RESUMABLE_STATUSES <= cr.RESUMABLE_STATUSES
    assert "learning_failed" in cr.RESUMABLE_STATUSES - cr.AUTO_RESUMABLE_STATUSES
    assert not cr.BLOCKED_STATUSES & cr.RESUMABLE_STATUSES
    assert cr.ATTEMPT_LEARNING_INTERRUPTED in cr.ATTEMPT_STATES
    assert (
        cr.FAILURE_CHECKPOINT_MISMATCH,
        cr.FAILURE_CHECKPOINT_CORRUPT,
        cr.FAILURE_CHECKPOINT_NOT_QUIESCENT,
        cr.FAILURE_DISK_FULL,
        cr.RECOVERY_RESUME_FROM_CHECKPOINT,
    ) == (
        "checkpoint_mismatch",
        "checkpoint_corrupt",
        "checkpoint_not_quiescent",
        "disk_full",
        "resume_from_checkpoint",
    )
    assert {"_hosts", "_supervise"} <= cr.NON_CELL_DIRS
    assert cr.MERGED_MARKER == "_merged.json"


def test_checkpoint_ids_and_ordinals():
    from evals.skill_bench.cell_records import (
        checkpoint_id,
        checkpoint_ordinal,
        parse_checkpoint_id,
    )

    assert checkpoint_id("init", 0) == "c0000-init"
    assert checkpoint_id("played", 7) == "c0007-played"
    assert checkpoint_id("done", 12345) == "c12345-done"
    assert [
        checkpoint_ordinal(k, p)
        for k, p in [
            ("init", 0),
            ("played", 0),
            ("done", 0),
            ("played", 7),
            ("done", 7),
        ]
    ] == [0, 1, 2, 15, 16]
    assert parse_checkpoint_id("c0007-played") == ("played", 7)
    for bad in (
        "c7-played",
        "c0001-init",
        "c0001-other",
        "LATEST",
        "c0001-done.partial",
    ):
        with pytest.raises(ValueError):
            parse_checkpoint_id(bad)
    for kind, position in (("init", 1), ("other", 0), ("done", -1)):
        with pytest.raises(ValueError):
            checkpoint_ordinal(kind, position)


def test_the_host_record_and_its_identity(monkeypatch):
    from evals.skill_bench import cell_records as cr

    host = cr.host_info()
    assert set(host) == {
        "hostname",
        "machine_id",
        "system",
        "release",
        "machine",
        "python",
        "platform",
    }
    # This machine (macOS or Linux) has an id: a 32-digit hash, never the raw id.
    assert host["machine_id"] and len(str(host["machine_id"])) == 32
    assert cr.machine_id() == host["machine_id"]
    assert cr.same_host(host, dict(host))
    assert cr.same_host(host, {**host, "release": "other kernel", "python": "3.99"})
    assert not cr.same_host(host, {**host, "machine": "other"})
    assert not cr.same_host(host, {**host, "machine_id": "f" * 32})
    # A Mac's hostname follows its network; the machine id names the host.
    assert cr.same_host(host, {**host, "hostname": "laptop-2.local"})
    # Records without a machine id (older ones) still compare the hostname.
    legacy = {k: v for k, v in host.items() if k != "machine_id"}
    assert cr.same_host(legacy, host)
    assert not cr.same_host(legacy, {**host, "hostname": "elsewhere"})
    assert not cr.same_host(None, host) and not cr.same_host({}, {})
    assert cr.host_key(host) == f"id:{host['machine_id']}"
    assert cr.host_key(legacy) == f"hostname:{host['hostname']}"
    assert str(host["machine_id"])[:8] in cr.host_text(host)


def test_the_machine_id_is_read_from_the_os(tmp_path, monkeypatch):
    """macOS's IOPlatformUUID (ioreg), Linux's machine-id; None when neither
    can be read, so the hostname is compared instead."""
    import subprocess

    from evals.skill_bench import cell_records as cr

    ioreg = (
        "+-o Mac  <class IOPlatformExpertDevice, id 0x100000001>\n"
        '    "IOPlatformUUID" = "1a2b3c4d-0000-4000-8000-00000000abcd"\n'
    )

    def fake_run(argv, **_kw):
        assert argv[0] == cr.IOREG
        return subprocess.CompletedProcess(argv, 0, stdout=ioreg, stderr="")

    monkeypatch.setattr(cr.subprocess, "run", fake_run)
    assert cr._raw_machine_id("Darwin") == "1A2B3C4D-0000-4000-8000-00000000ABCD"
    ioreg = "no platform device\n"
    assert cr._raw_machine_id("Darwin") is None
    machine_id = tmp_path / "machine-id"
    machine_id.write_text("0123456789abcdef0123456789abcdef\n")
    monkeypatch.setattr(
        cr, "LINUX_MACHINE_ID_FILES", (str(tmp_path / "missing"), str(machine_id))
    )
    assert cr._raw_machine_id("Linux") == "0123456789abcdef0123456789abcdef"
    machine_id.write_text("")
    assert cr._raw_machine_id("Linux") is None
    assert cr._raw_machine_id("Windows") is None

    def broken(argv, **_kw):
        raise OSError("no ioreg")

    monkeypatch.setattr(cr, "IOREG_RETRY_WAIT_S", 0.0)
    monkeypatch.setattr(cr.subprocess, "run", broken)
    assert cr._raw_machine_id("Darwin") is None


def test_a_failed_ioreg_is_tried_again_and_only_an_id_is_cached(monkeypatch):
    """One ``ioreg`` that timed out must not leave the whole process (every record it
    writes) without a machine id, compared by hostname, where a later Mac rename would
    refuse the resume. A failed run (timeout, could not start, non-zero exit, nothing
    printed) is tried again, up to IOREG_TRIES, with the longer IOREG_TIMEOUT_S; one that
    answered without an id, or a machine without ioreg, is not; and machine_id() caches
    only an id it read, so the next call asks again after a failure."""
    import subprocess

    from evals.skill_bench import cell_records as cr

    ioreg = '    "IOPlatformUUID" = "1a2b3c4d-0000-4000-8000-00000000abcd"\n'
    script: list[Any] = []
    seen: list[float] = []

    def fake_run(argv, *, timeout, **_kw):
        assert argv[0] == cr.IOREG
        seen.append(timeout)
        step = script.pop(0)
        if isinstance(step, BaseException):
            raise step
        code, out = step
        return subprocess.CompletedProcess(argv, code, stdout=out, stderr="")

    monkeypatch.setattr(cr, "IOREG_RETRY_WAIT_S", 0.0)
    monkeypatch.setattr(cr.subprocess, "run", fake_run)
    expected = "1A2B3C4D-0000-4000-8000-00000000ABCD"

    script[:] = [subprocess.TimeoutExpired("ioreg", 10), (0, ioreg)]
    assert cr._raw_machine_id("Darwin") == expected
    assert seen == [cr.IOREG_TIMEOUT_S] * 2 and cr.IOREG_TIMEOUT_S > 10
    script[:] = [OSError(11, "Resource temporarily unavailable"), (1, ""), (0, ioreg)]
    assert cr._raw_machine_id("Darwin") == expected and not script
    script[:] = [subprocess.TimeoutExpired("ioreg", 10)] * cr.IOREG_TRIES
    assert cr._raw_machine_id("Darwin") is None and not script
    # Nothing to gain from asking again.
    script[:] = [(0, "+-o Root  <class IOPlatformExpertDevice>\n"), (0, ioreg)]
    assert cr._raw_machine_id("Darwin") is None and len(script) == 1
    script[:] = [FileNotFoundError(2, "No such file"), (0, ioreg)]
    assert cr._raw_machine_id("Darwin") is None and len(script) == 1

    # machine_id(): a failure is not cached for good, the id is (and never asked for
    # again). After a failure the machine is not asked again for a while: a hung
    # ioreg would stall every record.
    monkeypatch.setattr(cr, "_machine_id", None)
    monkeypatch.setattr(cr, "_machine_id_failed_at", None)
    monkeypatch.setattr(cr.platform, "system", lambda: "Darwin")
    clock = [1000.0]
    monkeypatch.setattr(cr, "_hold_clock", lambda: clock[0])
    script[:] = [subprocess.TimeoutExpired("ioreg", 10)] * cr.IOREG_TRIES
    assert cr.machine_id() is None
    script[:] = [(0, ioreg)]
    clock[0] += cr.MACHINE_ID_FAILURE_HOLD_S - 1
    assert cr.machine_id() is None and len(script) == 1  # held: ioreg not run
    clock[0] += 1
    first = cr.machine_id()
    assert first is not None and len(first) == 32 and expected not in first
    assert cr.machine_id() == first and cr.host_info()["machine_id"] == first
    assert not script  # cached: no ioreg run after the first success


def test_discarded_segment_costs_price_a_call_that_finished_after_its_resume():
    """A discarded.jsonl row keeps what the ledger held at its resume; a
    call in flight then (a pause held it) gets its final line later. Each segment's cost
    is read from the final lines now: the late call counted, a call still without one
    counted apart, the row's own figure kept beside it."""
    from evals.skill_bench.cell_records import discarded_segment_costs

    rows = [
        {
            "segment": 1,
            "calls": [
                {"call_id": "a", "role": "actor", "final_line": True},
                {"call_id": "held", "role": "embedding", "final_line": False},
                {"call_id": "gone", "role": "mirix", "final_line": False},
            ],
            "cost_total": 0.01,
        },
        {
            "segment": 2,
            "calls": [{"call_id": "cut", "role": "actor", "final_line": True}],
            "cost_total": 0.0,
        },
        {"segment": 3, "calls": []},
    ]
    ledger = [
        {"event": "call_started", "call_id": "a"},
        _call("a", "actor", "learning"),
        {"event": "resume", "segment": 1},
        _call("held", "embedding", "learning", cost_total=0.005),
        _deadline("cut", "actor", "learning", unknown_cost_attempts=1),
        _call("live", "actor", "learning", cost_total=9.0),  # not discarded
    ]
    first, second, third = discarded_segment_costs(rows, ledger)
    assert first == {
        "segment": 1,
        "calls": 3,
        "cost_total": pytest.approx(0.015),
        "unknown_cost_calls": 0,
        "unfinished_calls": 1,
        "by_role": {"actor": 1, "embedding": 1, "mirix": 1},
        "cost_total_at_resume": 0.01,
        "finished_after_resume": 1,
    }
    assert (second["cost_total"], second["unknown_cost_calls"]) == (0.0, 1)
    assert second["finished_after_resume"] == 0
    assert third["calls"] == 0 and third["cost_total_at_resume"] is None
    assert discarded_segment_costs([], ledger) == []


def test_ledger_lines_carry_their_byte_offsets(tmp_path):
    from evals.skill_bench.cell_records import read_ledger_lines

    path = tmp_path / "gateway.jsonl"
    first = json.dumps({"event": "phase", "phase": "learning"}) + "\n"
    torn = '{"event": "call_sta'
    second = (
        json.dumps(
            {"event": "checkpoint", "id": "c0000-init", "note": "é"}, ensure_ascii=False
        )
        + "\n"
    )
    path.write_bytes((first + torn + "\n" + second).encode("utf-8"))
    lines = read_ledger_lines(path)
    assert [line.row["event"] for line in lines] == ["phase", "checkpoint"]
    offset = lines[1].offset
    assert offset == len((first + torn + "\n").encode("utf-8"))
    with path.open("rb") as handle:
        handle.seek(offset)
        assert json.loads(handle.readline())["id"] == "c0000-init"
    assert read_ledger_lines(tmp_path / "missing.jsonl") == []


def test_segment_calls_are_the_calls_started_between_the_two_markers():
    from evals.skill_bench.cell_records import LedgerLine, segment_calls

    def started(offset: int, call_id: str) -> LedgerLine:
        return LedgerLine(
            offset,
            {
                "event": "call_started",
                "call_id": call_id,
                "role": "actor",
                "item": "t1",
                "run_phase": "learning",
                "gateway_start": "g1",
            },
        )

    lines = [
        started(5, "before"),
        LedgerLine(10, {"event": "checkpoint", "id": "c0001-done"}),
        started(20, "in"),
        LedgerLine(
            25, {"event": "call", "call_id": "in", "status": "ok", "cost_total": 0.5}
        ),
        started(30, "flying"),
        LedgerLine(40, {"event": "resume", "segment": 1}),
        LedgerLine(
            45,
            {
                "event": "call",
                "call_id": "flying",
                "status": "error",
                "cost_total": 0.25,
                "unknown_cost_attempts": 1,
            },
        ),
        started(50, "after"),
    ]
    calls = segment_calls(lines, checkpoint_offset=10, resume_offset=40)
    assert [c["call_id"] for c in calls] == ["in", "flying"]
    assert calls[0] | {} == {
        "call_id": "in",
        "role": "actor",
        "item": "t1",
        "phase": "learning",
        "started_offset": 20,
        "gateway_start": "g1",
        "final_line": True,
        "status": "ok",
        "cost_total": 0.5,
        "unknown_cost": False,
    }
    assert calls[1]["unknown_cost"] is True and calls[1]["final_line"] is True
    everything = segment_calls(lines, checkpoint_offset=None, resume_offset=40)
    assert [c["call_id"] for c in everything] == ["before", "in", "flying"]
    assert (
        segment_calls(lines, checkpoint_offset=10, resume_offset=40, exclude={"in"})[0][
            "call_id"
        ]
        == "flying"
    )


def test_discarded_calls_are_summed_apart_and_kept_in_the_totals(tmp_path):
    rows = [
        _call("a", "actor", "learning"),
        _call("b", "judge", "learning"),
        _call("c", "actor", "learning"),
    ]
    path = _write(tmp_path / "gateway.jsonl", rows)
    summary = summarize_ledger(path, discarded={"a", "b"})
    assert summary["totals"]["calls"] == 3
    assert summary["discarded"]["calls"] == 2
    assert summary["discarded"]["cost_total"] == pytest.approx(0.02)
    assert summary["discarded"]["by_role"] == {"actor": 1, "judge": 1}
    assert summarize_ledger(path)["discarded"]["calls"] == 0


def test_discarded_rows_and_call_ids(tmp_path):
    from evals.skill_bench.cell_records import (
        AttemptPaths,
        discarded_call_ids,
        read_discarded,
    )

    paths = AttemptPaths(tmp_path / "attempt-01", 1)
    assert read_discarded(paths) == [] and discarded_call_ids(paths) == frozenset()
    paths.root.mkdir()
    paths.discarded_log.write_text(
        json.dumps({"segment": 1, "calls": [{"call_id": "x"}, {"call_id": "y"}]})
        + "\n"
        + json.dumps({"segment": 2, "calls": [{"call_id": "z"}]})
        + "\n"
    )
    assert [r["segment"] for r in read_discarded(paths)] == [1, 2]
    assert discarded_call_ids(paths) == {"x", "y", "z"}
    assert paths.discarded_dir(3) == paths.root / "discarded" / "seg-03"
    assert paths.checkpoint("c0000-init") == paths.root / "checkpoints" / "c0000-init"
    assert paths.latest_checkpoint == paths.root / "checkpoints" / "LATEST"


def test_cell_progress_reports_the_checkpoint_and_the_scored_items(tmp_path):
    from evals.skill_bench.cell_records import (
        CellPaths,
        cell_progress,
        write_json_atomic,
    )

    paths = CellPaths(tmp_path / "cell")
    assert cell_progress(paths)["attempt"] is None
    attempt = paths.attempt(2)
    attempt.root.mkdir(parents=True)
    write_json_atomic(
        attempt.state, {"state": "learning_interrupted", "learning": {"done": 4}}
    )
    attempt.checkpoints.mkdir()
    write_json_atomic(attempt.latest_checkpoint, {"id": "c0003-done", "ordinal": 8})
    attempt.results.write_text(
        json.dumps({"raw_id": "a", "score": {"primary": 1}})
        + "\n"
        + json.dumps({"raw_id": "b", "score": None})
        + "\n"
    )
    assert cell_progress(paths) == {
        "attempt": 2,
        "state": "learning_interrupted",
        "checkpoint_ordinal": 8,
        "learning_done": 4,
        "test_scored": 1,
    }


def test_ledger_paths_are_looked_up_relative_first(tmp_path):
    from evals.skill_bench.cell_records import resolve_blob_dir, resolve_ledger_path

    runs = tmp_path / "runs"
    run_id = "cell-a01-feed"
    state = {
        "run_id": run_id,
        "ledger": str(tmp_path / "elsewhere" / run_id / "gateway.jsonl"),
    }
    # Nothing exists: the relative path (the caller reports it missing).
    assert resolve_ledger_path(state, runs_dir=runs) == runs / run_id / "gateway.jsonl"
    # A recorded absolute path is used only when its directory is the run's.
    recorded = Path(state["ledger"])
    recorded.parent.mkdir(parents=True)
    recorded.write_text("")
    assert resolve_ledger_path(state, runs_dir=runs) == recorded
    stale = {**state, "ledger": str(tmp_path / "other-run" / "gateway.jsonl")}
    (tmp_path / "other-run").mkdir()
    (tmp_path / "other-run" / "gateway.jsonl").write_text("")
    assert resolve_ledger_path(stale, runs_dir=runs) == runs / run_id / "gateway.jsonl"
    # A merged view of several hosts: _hosts/<name>/<run>/gateway.jsonl.
    mirrored = runs / "_hosts" / "laptop-2" / run_id / "gateway.jsonl"
    mirrored.parent.mkdir(parents=True)
    mirrored.write_text("")
    assert resolve_ledger_path(state, runs_dir=runs) == mirrored
    # Another gateway runs directory beats the mirror, and the local one beats all.
    gateway_runs = tmp_path / "gw"
    (gateway_runs / run_id).mkdir(parents=True)
    (gateway_runs / run_id / "gateway.jsonl").write_text("")
    assert resolve_ledger_path(
        state, runs_dir=runs, gateway_runs_dirs=[gateway_runs]
    ) == (gateway_runs / run_id / "gateway.jsonl")
    local = runs / run_id / "gateway.jsonl"
    local.parent.mkdir(parents=True)
    local.write_text("")
    assert (
        resolve_ledger_path(state, runs_dir=runs, gateway_runs_dirs=[gateway_runs])
        == local
    )
    # Blobs: the runs directory's own first, then the recorded one.
    assert (
        resolve_blob_dir({"blob_dir": str(tmp_path / "gone")}, runs) == runs / "_blobs"
    )
    (tmp_path / "recorded_blobs").mkdir()
    assert resolve_blob_dir({"blob_dir": str(tmp_path / "recorded_blobs")}, runs) == (
        tmp_path / "recorded_blobs"
    )
    (runs / "_blobs").mkdir()
    assert resolve_blob_dir({"blob_dir": str(tmp_path / "recorded_blobs")}, runs) == (
        runs / "_blobs"
    )


# --- refusals and what an attempt went on despite -----------------------------------------


def _refused(call_id: str, role: str, phase: str, http: int, message: str, **kw: Any):
    """A final ledger line of a call the upstream (or the gateway) refused, as the
    gateway writes it."""
    source = kw.pop("source", "upstream")
    return _call(
        call_id,
        role,
        phase,
        status="upstream_error" if source == "upstream" else "rejected",
        http_status=http,
        cost_total=0.0,
        error={"source": source, "code": http, "message": message, **kw},
    )


def test_summarize_ledger_counts_refusals(tmp_path):
    rows = [
        _call("ok1", "learner", "learning"),
        _refused(
            "a",
            "learner",
            "learning",
            413,
            "Request payload too large",
        ),
        _refused(
            "b",
            "learner",
            "learning",
            413,
            "Request payload too large",
        ),
        _refused("c", "judge", "learning", 400, "Provider returned error"),
        _refused("d", "actor", "test", 404, "No endpoints found that support tool use"),
        _refused(
            "e",
            "mirix",
            "learning",
            409,
            "phase mismatch",
            source="gateway",
            type="phase_mismatch",
        ),
        # Not refusals: an account refusal, a plain 404, infrastructure.
        _refused(
            "f", "actor", "test", 404, "No endpoints found matching your data policy"
        ),
        _refused("g", "actor", "test", 404, "Not Found"),
        _call(
            "h",
            "actor",
            "test",
            status="upstream_error",
            http_status=502,
            error={"type": "upstream_infra_failure", "message": "bad gateway"},
        ),
    ]
    path = _write(tmp_path / "gateway.jsonl", rows)
    summary = summarize_ledger(path, discarded={"a"})
    assert summary["refusals"] == {
        "learning/judge": {"payload/400": 1},
        "learning/learner": {"payload/413": 2},  # the discarded call counts too
        "learning/mirix": {"gateway/409": 1},
        "test/actor": {"no_endpoint/404": 1},
    }
    clean = _write(tmp_path / "clean.jsonl", [_call("x", "actor", "test")])
    assert summarize_ledger(clean)["refusals"] == {}


def test_consecutive_rollbacks_count_trailing_resumes_of_one_checkpoint():
    from evals.skill_bench.cell_records import consecutive_rollbacks

    def resume(
        cid: str | None, kind: str | None = "infra", code: str = "mirix_learning"
    ):
        failure = None if kind is None else {"kind": kind, "code": code}
        return {"checkpoint": cid, "failure": failure}

    rows = [
        resume("c0003-played"),
        resume("c0003-played", None),  # an interruption breaks the run
        resume("c0003-played"),
        resume("c0003-played"),
        resume("c0003-played"),
    ]
    count = lambda r, cid="c0003-played": consecutive_rollbacks(  # noqa: E731
        r, cid, code="mirix_learning"
    )
    assert count(rows) == 3
    assert count(rows[:1]) == 1 and count(rows[:2]) == 0
    assert count([*rows, resume("c0004-played")]) == 0  # another checkpoint
    assert count([*rows, resume("c0004-played")], "c0004-played") == 1
    assert count([*rows, resume("c0003-played", "harness")]) == 0
    assert count([*rows, resume("c0003-played", code="upstream_infra")]) == 0
    assert count([]) == 0 and count(None) == 0  # type: ignore[arg-type]
    assert count([*rows, "not a row"]) == 0  # type: ignore[list-item]


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _unserved(status: int) -> dict[str, Any]:
    return {
        "kind": "harness",
        "code": "provider_unserved",
        "role": "actor",
        "status": status,
        "detail": "refused",
        "retries": 0,
    }


def test_tolerated_failures_reads_rows_snapshot_costs_and_attempt(tmp_path):
    from evals.skill_bench.cell_records import (
        AttemptPaths,
        tolerated_failures,
        write_json_atomic,
    )

    empty = AttemptPaths(tmp_path / "none", 1)
    assert tolerated_failures(empty) == {
        "learning": {"provider_unserved": [], "retrieval_degraded": []},
        "test": {"provider_unserved": [], "retrieval_degraded": []},
        "method": {"counters": {}, "mirix": None},
        "refusals": {},
        "accepted_learning_calls": [],
    }
    paths = AttemptPaths(tmp_path / "attempt-01", 1)
    _jsonl(
        paths.learning,
        [
            {"position": 0, "raw_id": "a", "task_uid": "ua", "score": {"primary": 1}},
            {
                "position": 1,
                "raw_id": "b",
                "task_uid": "ub",
                "score": {"primary": 0.0},
                "end_reason": "provider_unserved",
                "provider_unserved": True,
                "failure": _unserved(413),
            },
            {"position": 2, "raw_id": "c", "task_uid": "uc", "score": {"primary": 0}},
            {  # position 2 replayed: its latest row counts
                "position": 2,
                "raw_id": "c",
                "task_uid": "uc",
                "score": {"primary": 0.5},
                "retrieval_degraded": True,
            },
        ],
    )
    _jsonl(
        paths.results,
        [
            {
                "raw_id": "t1",
                "score": None,
                "failure": {"kind": "harness", "code": "upstream_rejected"},
            },
            {
                "raw_id": "t1",
                "score": {"primary": 0.0},
                "end_reason": "provider_unserved",
                "provider_unserved": True,
                "failure": _unserved(404),
            },
            {"raw_id": "t2", "score": {"primary": 1.0}, "retrieval_degraded": True},
            {"raw_id": "t3", "score": {"primary": 1.0}, "retrieval_degraded": True},
            {"raw_id": "t3", "score": {"primary": 1.0}},  # its latest scored row: clean
            {"raw_id": "t4", "score": None, "retrieval_degraded": True},  # unscored
        ],
    )
    tolerated = {"counts": {"retrievals_degraded": 1}, "problems": []}
    write_json_atomic(
        paths.frozen_snapshot,
        {
            "sha256": "0" * 64,
            "snapshot": {
                "counters": {
                    "induction_rejected": 2,
                    "inductions": 9,
                    "judge_rejected": 0,
                },
                "tolerated": tolerated,
            },
        },
    )
    write_json_atomic(
        paths.costs, {"refusals": {"learning/learner": {"payload/413": 2}}}
    )
    accepted = [{"call_id": "g1", "role": "mirix", "http_status": 502}]
    write_json_atomic(paths.state, {"accepted_learning_calls": accepted})
    assert tolerated_failures(paths) == {
        "learning": {
            "provider_unserved": [
                {
                    "position": 1,
                    "raw_id": "b",
                    "task_uid": "ub",
                    "status": 413,
                    "class": "payload",
                }
            ],
            "retrieval_degraded": [{"position": 2, "raw_id": "c", "task_uid": "uc"}],
        },
        "test": {"provider_unserved": ["t1"], "retrieval_degraded": ["t2"]},
        "method": {"counters": {"induction_rejected": 2}, "mirix": tolerated},
        "refusals": {"learning/learner": {"payload/413": 2}},
        "accepted_learning_calls": accepted,
    }


def test_tolerated_counter_keys_match_the_methods():
    from evals.skill_bench import cell_records as C
    from evals.skill_bench.methods import awm, mirix, reasoningbank

    assert set(C.TOLERATED_COUNTER_KEYS) == set(awm.REFUSAL_COUNTERS) | set(
        reasoningbank.REFUSAL_COUNTERS
    )
    assert C.METHOD_INDUCTIONS_FILE == awm.INDUCTIONS_FILE
    assert C.METHOD_EXTRACTIONS_FILE == reasoningbank.EXTRACTIONS_FILE
    assert C.METHOD_MIRIX_LEARNING_FILE == mirix.LEARNING_RECORD_FILE
    assert {mirix.TOLERATED_REFUSED, mirix.TOLERATED_ROLLED_BACK} == {
        "gateway_refusal",
        "rolled_back",
    }


def test_recorded_problems_counts_what_an_attempt_went_on_despite(tmp_path):
    from evals.skill_bench.cell_records import AttemptPaths, recorded_problems

    paths = AttemptPaths(tmp_path / "attempt-02", 2)
    assert recorded_problems(paths) == {}  # nothing there: never raises
    _jsonl(
        paths.failures,
        [
            {"code": "provider_unserved", "recovery": "score_and_continue"},
            {"code": "provider_unserved", "recovery": "score_and_continue"},
            {"code": "upstream_refusal_streak", "recovery": "restart_cell"},
            {"code": "mirix_learning", "recovery": "accept_failed_batch"},
            {"code": "upstream_infra", "recovery": "resume_from_checkpoint"},
        ],
    )
    _jsonl(
        paths.learning, [{"position": 0, "retrieval_degraded": True}, {"position": 1}]
    )
    _jsonl(paths.results, [{"raw_id": "a", "retrieval_degraded": True}])
    _jsonl(
        paths.method_dir / "inductions.jsonl",
        [
            {
                "event": "induction",
                "retry": {"refusal": {}},
                "rejected": {"status": 413},
            },
            {"event": "induction", "retry": {"refusal": {}}},
            {"event": "induction"},
            {"event": "judge_refused", "task_uid": "u"},
        ],
    )
    _jsonl(
        paths.method_dir / "extractions.jsonl",
        [
            {"event": "extraction", "refused": {"status": 413}},
            {"event": "extraction", "refused": None},
            {"event": "judge_refused"},
        ],
    )
    _jsonl(
        paths.method_dir / "mirix_learning.jsonl",
        [
            {"event": "tolerated", "reason": "gateway_refusal"},
            {"event": "tolerated", "reason": "rolled_back"},
            {"event": "tolerated", "reason": "rolled_back"},
            {"event": "close"},
        ],
    )
    assert recorded_problems(paths) == {
        "provider_unserved": 2,
        "upstream_refusal_streak": 1,
        "mirix_batch_accepted": 1,
        "retrieval_degraded": 2,
        "induction_rejected": 1,
        "induction_retries": 2,
        "extraction_rejected": 1,
        "judge_rejected": 2,
        "mirix_gateway_refusal": 1,
        "mirix_rolled_back": 2,
    }
    # A file that cannot be read counts 0 (a directory where a file should be).
    paths.failures.unlink()
    paths.failures.mkdir()
    assert "provider_unserved" not in recorded_problems(paths)


def test_the_removed_data_directory_is_not_a_cell():
    from evals.skill_bench.cell_records import NON_CELL_DIRS

    assert "_removed" in NON_CELL_DIRS
    assert {"_gateway", "_blobs", "_hosts", "_supervise"} <= NON_CELL_DIRS


def test_git_info_leaves_the_frozen_protocol_record_out_of_dirty(tmp_path):
    import subprocess

    from evals.skill_bench.cell_records import GIT_DIRTY_EXCLUDED, git_info

    assert GIT_DIRTY_EXCLUDED == ("evals/skill_bench/config/protocol_frozen.json",)
    repo = tmp_path / "repo"
    record = repo / "evals" / "skill_bench" / "config" / "protocol_frozen.json"
    code = repo / "evals" / "skill_bench" / "run.py"
    record.parent.mkdir(parents=True)
    record.write_text("{}\n")
    code.write_text("x = 1\n")

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
            cwd=repo,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    clean = git_info(repo)
    assert clean["dirty"] is False and clean["commit"]
    record.write_text('{"auto_freeze": {}}\n')  # a run froze the protocol
    assert git_info(repo)["dirty"] is False
    # From a subdirectory too (the pathspecs are the repository's top).
    assert git_info(repo / "evals")["dirty"] is False
    code.write_text("x = 2\n")
    assert git_info(repo)["dirty"] is True
    assert git_info(repo / "evals")["dirty"] is True


def test_git_info_counts_untracked_files_a_version_hashes_as_dirty(tmp_path):
    """The fingerprint and the code digests walk the disk, so a new prompt or
    module that was never ``git add``-ed changes the version, and the commit alone does
    not rebuild it. Untracked files under the paths a version segment's tree diff covers
    (``TREE_PATHSPECS_HARNESS``/``_MIRIX``) make the checkout dirty (so run.py captures
    them in ``tree.diff``); untracked files elsewhere, under the excluded tests, runs and
    data, or gitignored, do not."""
    import subprocess

    from evals.skill_bench.cell_records import (
        TREE_PATHSPECS_HARNESS,
        TREE_PATHSPECS_MIRIX,
        git_info,
        tree_pathspecs,
    )

    assert tree_pathspecs(mirix=False) == list(TREE_PATHSPECS_HARNESS)
    assert tree_pathspecs(mirix=True) == [
        *TREE_PATHSPECS_HARNESS,
        *TREE_PATHSPECS_MIRIX,
    ]
    assert "scripts/start_server.py" in TREE_PATHSPECS_MIRIX
    repo = tmp_path / "repo"
    bench = repo / "evals" / "skill_bench"
    (bench / "methods" / "prompts").mkdir(parents=True)
    (bench / "run.py").write_text("x = 1\n")
    (repo / ".gitignore").write_text("*.log\n")

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
            cwd=repo,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    assert git_info(repo)["dirty"] is False
    assert git_info(repo)["untracked"] == 0
    # Not in a version: elsewhere, the excluded directories, a gitignored file.
    for rel in (
        "notes.txt",
        "evals/skill_bench/tests/test_new.py",
        "evals/skill_bench/runs/cell/x.json",
        "evals/skill_bench/data/x.json",
        "mirix/tests/test_x.py",
        "evals/skill_bench/methods/prompts/debug.log",
    ):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text("x\n")
    assert (git_info(repo)["dirty"], git_info(repo)["untracked"]) == (False, 0)
    # A new prompt only: dirty, from the top and from a subdirectory.
    prompt = bench / "methods" / "prompts" / "awm" / "new_prompt.json"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("{}\n")
    assert protocol.is_fingerprinted("methods/prompts/awm/new_prompt.json")
    assert (git_info(repo)["dirty"], git_info(repo)["untracked"]) == (True, 1)
    assert git_info(repo / "evals")["dirty"] is True
    prompt.unlink()
    # A new MIRIX module (MIRIX cells hash the package).
    (repo / "mirix" / "agent").mkdir(parents=True)
    (repo / "mirix" / "agent" / "new.py").write_text("y = 1\n")
    assert (git_info(repo)["dirty"], git_info(repo)["untracked"]) == (True, 1)
    (repo / "mirix" / "agent" / "new.py").unlink()
    # The frozen record's side files (the lock file where the record's directory cannot
    # be locked, and the temporary file of its atomic replace) are never part
    # of a version: a freeze, which reads git_info under the lock, records a clean tree.
    from evals.skill_bench.tools import freeze_protocol

    record = repo / "evals" / "skill_bench" / "config" / "protocol_frozen.json"
    record.parent.mkdir(parents=True)
    freeze_protocol.lock_file_path(record).write_text("")
    record.with_name(f".{record.name}.tmp").write_text("{}\n")
    assert (git_info(repo)["dirty"], git_info(repo)["untracked"]) == (False, 0)
    # Outside a repository: unknown, not clean.
    (tmp_path / "plain").mkdir()
    unknown = git_info(tmp_path / "plain")
    assert unknown["dirty"] is None and unknown["untracked"] is None
