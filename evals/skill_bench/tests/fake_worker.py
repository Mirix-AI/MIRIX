"""Stands in for ``run.py`` in scheduler tests: ``python -m evals.skill_bench.tests.fake_worker``.

It accepts run.py's command line, behaves like a cell process towards the scheduler (holds
``cell.lock`` for its lifetime, writes ``process.json`` and ``status.json``, exits with the
status's code) and makes no model call. ``<out>/behavior.json`` sets what it does:

* ``status``: the final status (default ``complete``);
* ``sleep_s``: how long it runs (default 0.3);
* ``until``: a file it waits for before finishing (a test releases it);
* ``statuses``: the final status of each successive launch (the last one repeats);
* ``exit_code``: the exit code, when not the status's (``cell_records.EXIT_CODES``);
* ``status_extra``: more fields of the final ``status.json`` (``failure_kind``);
* ``progress``: what it reports as its progress after each successive launch (the last
  one repeats), written to ``<out>/progress.json`` for a test's ``cell_progress``;
* ``die``: exit with this code at once after writing its ``running`` status, leaving it
  behind (a worker killed mid-run); ``crash_at_start``: exit with this code before
  writing anything (an import error); either may be a list per launch (null: go on);
* ``ignore_sigterm``: note a ``SIGTERM`` and go on (a worker still finishing a
  checkpoint); otherwise ``SIGTERM`` ends it ``interrupted`` (130) with ``signal:
  SIGTERM`` in its status, as ``run.py``;
* ``attempts``: keep ``attempt-NN/attempt.json`` as ``run.py`` does, for the scheduler's
  attempt reads: a launch with ``--new-attempt`` (or of a cell without an attempt)
  starts the next attempt at once (recording ``--note``), any other resumes the latest;
  the attempt's ``state`` follows the final status (``complete``, else ``testing``;
  ``attempt_state`` sets it).

It appends ``start`` / ``end`` / ``signal`` rows to ``<out>/../_fake_workers.jsonl`` (to
measure how many ran at once) and writes the provider-looking variables it can see to
``<out>/env.json``.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

from evals.skill_bench.cell_records import (
    EXIT_CODES,
    STATUS_BUSY,
    STATUS_INTERRUPTED,
    CellPaths,
    now,
    read_json,
    write_json_atomic,
)
from evals.skill_bench.infra.processes import (
    OwnedProcessError,
    OwnerLock,
    append_jsonl,
    process_start_time,
)


class _Terminated(Exception):
    """SIGTERM reached the fake worker (run.py cancels its main task)."""


def _nth(values: Any, n: int) -> Any:
    return values[min(n, len(values) - 1)] if isinstance(values, list) else values


def _start_attempt(paths: CellPaths, rest: list[str]) -> Path:
    """The attempt this launch runs, as run.py's plan step leaves it: a new one with
    ``--new-attempt`` (or none yet), else the latest. Returns its ``attempt.json``."""
    numbers = paths.attempt_numbers()
    latest = numbers[-1] if numbers else 0
    new = "--new-attempt" in rest or latest == 0
    number = latest + 1 if new else latest
    state = paths.attempt(number).state
    record = read_json(state) or {
        "attempt": number,
        "run_id": f"{paths.root.name}-a{number:02d}-fake",
        "state": "learning",
    }
    notes = [a.partition("=")[2] for a in rest if a.startswith("--note=")]
    if new and "--note" in rest:
        notes.append(rest[rest.index("--note") + 1])
    if new and notes:
        record["note"] = {"text": notes[-1]}
    write_json_atomic(state, record)
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gateway-url")
    parser.add_argument("--gateway-runs-dir")
    args, rest = parser.parse_known_args(argv)
    paths = CellPaths(args.out)
    paths.root.mkdir(parents=True, exist_ok=True)
    log = paths.root.parent / "_fake_workers.jsonl"
    try:
        lock = OwnerLock.acquire(paths.lock)  # retries across a probe, as run.py
    except OwnedProcessError:
        return EXIT_CODES[STATUS_BUSY]  # as run.py: another live harness has the cell
    behavior = read_json(paths.root / "behavior.json") or {}
    seen_path = paths.root / "launches_seen.json"
    seen = int((read_json(seen_path) or {}).get("n", 0))
    write_json_atomic(seen_path, {"n": seen + 1})
    crash = _nth(behavior.get("crash_at_start"), seen)
    if crash is not None:
        append_jsonl(log, {"cell": paths.root.name, "event": "crash", "t": time.time()})
        os._exit(int(crash))

    def on_sigterm(_signum: int, _frame: Any) -> None:
        append_jsonl(
            log, {"cell": paths.root.name, "event": "signal", "t": time.time()}
        )
        if not behavior.get("ignore_sigterm"):
            raise _Terminated()

    signal.signal(signal.SIGTERM, on_sigterm)
    signalled = False
    attempt_state = _start_attempt(paths, rest) if behavior.get("attempts") else None
    try:
        pid = os.getpid()
        write_json_atomic(
            paths.process,
            {"pid": pid, "started": process_start_time(pid), "argv": sys.argv},
        )
        write_json_atomic(
            paths.status, {"status": "running", "pid": pid, "updated_at": now()}
        )
        write_json_atomic(
            paths.root / "env.json",
            {
                "keys": sorted(
                    k for k in os.environ if k.endswith(("_API_KEY", "_TOKEN"))
                ),
                "gateway_url": args.gateway_url,
                "gateway_runs_dir": args.gateway_runs_dir,
                "argv": sys.argv[1:],
                "rest": rest,
            },
        )
        append_jsonl(
            log,
            {"cell": paths.root.name, "event": "start", "t": time.time(), "rest": rest},
        )
        die = _nth(behavior.get("die"), seen)
        if die is not None:
            os._exit(int(die))  # no final status: the worker "was killed"
        try:
            until = behavior.get("until")
            if until:
                deadline = time.monotonic() + float(behavior.get("until_timeout_s", 60))
                while not Path(until).exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
            time.sleep(float(behavior.get("sleep_s", 0.3)))
            status = str(behavior.get("status", "complete"))
            if behavior.get("statuses"):
                status = str(_nth(behavior["statuses"], seen))
        except _Terminated:
            status = STATUS_INTERRUPTED
            signalled = True
        if behavior.get("progress") is not None:
            write_json_atomic(
                paths.root / "progress.json", _nth(behavior["progress"], seen)
            )
        if attempt_state is not None:
            record = read_json(attempt_state) or {}
            record["state"] = behavior.get("attempt_state") or (
                "complete" if status == "complete" else "testing"
            )
            write_json_atomic(attempt_state, record)
        append_jsonl(log, {"cell": paths.root.name, "event": "end", "t": time.time()})
        code = int(behavior.get("exit_code", EXIT_CODES.get(status, 1)))
        if status == STATUS_INTERRUPTED:
            code = EXIT_CODES[STATUS_INTERRUPTED]
        write_json_atomic(
            paths.status,
            {
                "status": status,
                "exit_code": code,
                "pid": pid,
                "detail": "fake",
                "updated_at": now(),
                # As run.py: the signal that interrupted it.
                **({"signal": "SIGTERM"} if signalled else {}),
                **(behavior.get("status_extra") or {}),
            },
        )
        return code
    finally:
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
