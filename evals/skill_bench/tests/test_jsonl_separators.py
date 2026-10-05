"""JSONL readers split records on "\\n" only.

Every writer here uses ``json.dumps(..., ensure_ascii=False)``, so a model's text that holds
U+2028 / U+2029 / U+0085 reaches the file as the raw character (the ASCII control
characters ``str.splitlines()`` also splits on are escaped by ``json.dumps``).
``str.splitlines()`` treats those as line breaks: it would tear such a record into two
unparsable halves, which a tolerant reader drops silently (a result row gone from the
report) and a strict one raises on (a learning resume that cannot read its records).
These tests keep every reader on "\\n".
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from evals.skill_bench import cell_records, protocol, schedule
from evals.skill_bench.infra import failures, processes
from evals.skill_bench.infra import gateway as gw_mod
from evals.skill_bench.methods import awm
from evals.skill_bench.methods.reasoningbank import (
    BANK_FILE,
    EMBEDDINGS_FILE,
    ReasoningBank,
)
from evals.skill_bench.tools import sync_runs

#: Characters ``str.splitlines()`` splits on that ``json.dumps(ensure_ascii=False)`` writes
#: raw; it escapes the others (form feed, U+001C..U+001E) as ``\\f`` / ``\\u001c``.
SEPARATORS = "\u2028\u2029\u0085"
TEXT = f"before{SEPARATORS}after"


def write_rows(path: Path, rows: list[dict], *, separators: bool = True) -> None:
    for row in rows:
        processes.append_jsonl(path, row)
    if not separators:
        return
    raw = path.read_text(encoding="utf-8")
    # The writer keeps each character raw, as in production.
    assert all(char in raw for char in SEPARATORS)
    assert len(raw.splitlines()) > len(rows)  # and splitlines() would tear the records


def test_read_jsonl_counted_keeps_a_record_with_line_separators(tmp_path):
    path = tmp_path / "results.jsonl"
    rows = [{"raw_id": "1", "final_answer": TEXT}, {"raw_id": "2", "final_answer": "x"}]
    write_rows(path, rows)
    assert cell_records.read_jsonl_counted(path) == (rows, 0)


def test_failure_log_rows_keep_a_record_with_line_separators(tmp_path):
    path = tmp_path / "failures.jsonl"
    rows = [{"phase": "test", "detail": TEXT}, {"phase": "learning", "detail": "y"}]
    write_rows(path, rows)
    assert failures.FailureLog(path).rows() == rows


def test_registry_rows_keep_a_record_with_line_separators(tmp_path):
    path = tmp_path / "owned_processes.jsonl"
    write_rows(path, [{"event": "started", "pid": 4242, "argv": ["python", TEXT]}])
    assert processes._registry_rows(path)[4242]["argv"] == ["python", TEXT]


def test_starts_identity_reads_a_first_line_with_line_separators():
    line = json.dumps({"start_id": "abc", "note": TEXT}, ensure_ascii=False)
    assert sync_runs.starts_identity(line + "\n") == "abc"


def test_awm_pool_rows_with_line_separators_parse(tmp_path):
    # Only success rows are rebuilt (from their objects); any row must parse first.
    write_rows(tmp_path / awm.POOL_FILE, [{"event": "note", "instruction": TEXT}])
    assert awm.read_pool(tmp_path) == []


def test_reasoningbank_bank_with_line_separators_loads(tmp_path):
    write_rows(
        tmp_path / EMBEDDINGS_FILE,
        [{"id": "t1", "embedding": [3.0, 4.0]}],
        separators=False,
    )
    row = {
        "task_uid": "t1",
        "query": TEXT,
        "status": "success",
        "memory_items": [{"title": TEXT, "description": "d", "content": TEXT}],
        "embedding_sha256": "0" * 64,
        "embedding_dims": 2,
    }
    write_rows(tmp_path / BANK_FILE, [row])
    bank = ReasoningBank.__new__(ReasoningBank)
    bank._load_bank(tmp_path)
    assert [e.query for e in bank._entries] == [TEXT]
    assert bank._entries[0].memory_items[0].content == TEXT
    assert [round(x, 6) for x in bank._vectors[0]] == [0.6, 0.8]


def test_gateway_state_from_ledger_reads_a_record_with_line_separators(tmp_path):
    # A torn record is skipped (X15): the pinned model on it would be lost on a restart.
    gateway = gw_mod.Gateway(
        gw_mod.GatewayConfig(
            api_key="k", models=protocol.load_models(), runs_dir=tmp_path
        )
    )
    model = protocol.CHAT_MODELS[0]
    write_rows(
        tmp_path / "run-1" / "gateway.jsonl",
        [
            {
                "event": "call_started",
                "endpoint": gw_mod.CHAT,
                "model": model,
                "note": TEXT,
            },
            {"event": "phase", "phase": "test", "note": TEXT},
        ],
    )
    # A line that parses but is no record (a scalar) is skipped, not an AttributeError.
    with (tmp_path / "run-1" / "gateway.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("123\n")
    state = gateway._state_from_ledger("run-1")
    assert (state.model, state.phase, state.abandoned) == (model, "test", False)


def test_live_gateways_keeps_a_registry_record_with_line_separators(tmp_path):
    # This process stands in for a live gateway: same pid, same start time.
    registry = tmp_path / "_gateway" / "owned_processes.jsonl"
    row = {
        "event": "started",
        "name": "gateway",
        "pid": os.getpid(),
        "started": processes.process_start_time(os.getpid()),
        "owner_lock": "gateway.lock",
        "argv": ["python", TEXT],
    }
    write_rows(registry, [row])
    with registry.open("a", encoding="utf-8") as fh:
        fh.write("123\n")
    assert schedule.live_gateways(tmp_path) == [(row, registry.parent / "gateway.lock")]
