"""Stands in for ``schedule.py`` in supervisor tests: ``python -m evals.skill_bench.tests.fake_scheduler``.

It accepts schedule.py's command line (checked by schedule.py's own parser), launches no
cell and makes no call. ``<runs>/_fake_scheduler.json`` scripts its successive runs:
``{"runs": [step, ...]}`` (the last step repeats). A step may set:

* ``exit``: its exit code (default 0);
* ``pause_reason``: the ``pause_reason`` its ``state.json`` shows while it runs and at the
  end; ``attention`` / ``attention_reasons``: the cells it reports as needing a person;
* ``sleep_s``: how long it runs (default 0); ``until``: a file it waits for first;
* ``lock``: take ``_schedule/scheduler.lock`` like the real scheduler (default true; held:
  refuse with exit 4 and an ``{"error": ...}`` line on stderr);
* ``ignore_sigterm``: note ``SIGTERM`` and go on. Otherwise ``SIGTERM`` stops it with
  exit 3 and ``stopped``, as the real scheduler;
* ``no_final_state``: exit without the final ``state.json`` write (a crash);
* ``kill_self``: die by ``SIGKILL`` once started (a crash by a signal);
* ``crash_before_one_off``: exit with this code after taking the lock but before
  applying the one-off options (a crash in ``ensure_gateway``): its ``state.json`` never
  says ``one_off_applied``, which a real scheduler writes once it applied them;
* ``resource_wait`` / ``running`` / ``blocked`` / ``new_attempts``: what its
  ``state.json`` shows (the cell the resource gate holds; the running cells; the blocked
  cells; the ``--new-attempt-cells`` requests with their estimates).

Every run appends ``start`` / ``signal`` / ``end`` rows (argv, pid) to
``<runs>/_fake_scheduler.jsonl``.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

from evals.skill_bench import schedule
from evals.skill_bench.cell_records import now, read_json, write_json_atomic
from evals.skill_bench.infra.processes import (
    OwnedProcessError,
    OwnerLock,
    append_jsonl,
    process_start_time,
)


def main(argv: list[str] | None = None) -> int:
    args = schedule.build_parser().parse_args(argv)
    runs = Path(args.runs_dir).absolute()
    control = runs / "_schedule"
    control.mkdir(parents=True, exist_ok=True)
    log = runs / "_fake_scheduler.jsonl"
    script = read_json(runs / "_fake_scheduler.json") or {}
    count_path = runs / "_fake_scheduler_count.json"
    n = int((read_json(count_path) or {}).get("n", 0))
    write_json_atomic(count_path, {"n": n + 1})
    steps = script.get("runs") or [{}]
    step: dict[str, Any] = steps[min(n, len(steps) - 1)]
    pid = os.getpid()
    append_jsonl(
        log,
        {"event": "start", "n": n, "pid": pid, "argv": sys.argv[1:], "t": time.time()},
    )
    lock = None
    if step.get("lock", True):
        try:
            lock = OwnerLock.acquire(control / "scheduler.lock", wait_s=0.0)
        except OwnedProcessError:
            print(
                json.dumps({"error": f"another scheduler is running on {runs}"}),
                file=sys.stderr,
            )
            append_jsonl(log, {"event": "refused", "n": n, "pid": pid})
            return schedule.EXIT_REFUSED
    if step.get("crash_before_one_off") is not None:
        code = int(step["crash_before_one_off"])
        append_jsonl(log, {"event": "end", "n": n, "pid": pid, "code": code})
        if lock is not None:
            lock.release()
        return code
    stopped = False

    def on_sigterm(_signum: int, _frame: Any) -> None:
        nonlocal stopped
        append_jsonl(log, {"event": "signal", "n": n, "pid": pid, "t": time.time()})
        if not step.get("ignore_sigterm"):
            stopped = True

    signal.signal(signal.SIGTERM, on_sigterm)
    state = {
        "at": now(),
        "pid": pid,
        "started": process_start_time(pid),
        "pause_reason": step.get("pause_reason"),
        "needs_attention": step.get("attention") or {},
        "attention_reasons": step.get("attention_reasons") or {},
        "blocked": step.get("blocked") or {},
        "complete": [],
        "running": step.get("running") or [],
        "resource_wait": step.get("resource_wait"),
        "one_off_applied": True,
        "queued": [],
        "waiting": {},
        "budget": step.get("budget") or {},
        "new_attempts": step.get("new_attempts") or {},
        "exit_code": None,
    }
    write_json_atomic(control / "state.json", state)
    if step.get("kill_self"):
        os.kill(pid, signal.SIGKILL)
    try:
        deadline = time.monotonic() + float(step.get("sleep_s", 0))
        until = step.get("until")
        while not stopped:
            if until and not Path(until).exists():
                time.sleep(0.02)
                continue
            if time.monotonic() >= deadline:
                break
            time.sleep(0.02)
        code = schedule.EXIT_PAUSED if stopped else int(step.get("exit", 0))
        append_jsonl(log, {"event": "end", "n": n, "pid": pid, "code": code})
        if not step.get("no_final_state"):
            write_json_atomic(
                control / "state.json",
                {**state, "at": now(), "stopped": stopped, "exit_code": code},
            )
        return code
    finally:
        if lock is not None:
            lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
