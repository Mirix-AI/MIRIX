"""One ``run.py`` cell in a process of its own, for the tests that kill it.

``tests/test_resume_integration.py`` starts this module as a subprocess, SIGKILLs or
SIGTERMs it in the middle of a learning item, and starts it again to resume the cell.
It runs :func:`evals.skill_bench.run.run_cell` (the CLI's own signal handling and
outcome line) on a :class:`~evals.skill_bench.run.CellRunner` with the test seams:

* the benchmark is :class:`~evals.skill_bench.tests.run_support.TinyEnv`;
* the method is the registered one (``mirix``, ``awm``, ``reasoningbank``: the real
  classes), or ``storing``
  (:class:`~evals.skill_bench.tests.run_support.StoringMethod`, as ``reasoningbank``);
* MIRIX is this checkout's real server (``infra/mirix_server.py``) on the compose
  Postgres; ``--db-tag`` gives the test a cell database of its own
  (``sb_alfworld_luna_mirix_<tag>_a1_...``), never one a real or smoke cell uses;
* the gateway is the test's (``--gateway-url``, loopback, over the fake upstream);
* ``--salt`` fixes the cell's salt (two cell directories then have the same task uids,
  so their frozen memories can be compared);
* ``--code-file`` pins the code fingerprint to the one the first invocation recorded
  there: other work in the checkout between the kill and the resume is not a code change
  of the cell under test (the resume's same-code guard has tests of its own);
* ``--stall-checkpoint CID`` blocks the capture of that checkpoint for good, at
  ``--stall-point`` ``partial`` (everything is in ``CID.partial`` but its
  ``checkpoint.json``) or ``renamed`` (``CID`` is in place, ``LATEST`` does not name it
  yet), after writing ``--stall-flag``: the test kills the process there.

The outcome line goes to stdout and the exit code is the CLI's.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from evals.skill_bench import protocol
from evals.skill_bench import run as run_module
from evals.skill_bench.cell_records import (
    CHECKPOINT_RECORD,
    CellPaths,
    write_json_atomic,
)
from evals.skill_bench.infra.mirix_server import CellServerSpec, MirixServer
from evals.skill_bench.run import CellRunner, CellSpec, RunOptions, run_cell
from evals.skill_bench.tests.run_support import FIXED_GIT, StoringMethod, TinyEnv

METHODS = ("mirix", "awm", "reasoningbank", "storing")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evals.skill_bench.tests.resume_worker"
    )
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--out", type=Path, required=True, help="the cell directory")
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--gateway-runs-dir", type=Path, required=True)
    parser.add_argument("--train", type=int, default=6)
    parser.add_argument("--val", type=int, default=2)
    parser.add_argument("--test", type=int, default=2)
    parser.add_argument("--db-tag", help="MIRIX: a cell database of the test's own")
    parser.add_argument(
        "--salt", help="the cell's salt (written when the cell has none)"
    )
    parser.add_argument("--code-file", type=Path, help="pin the code fingerprint here")
    parser.add_argument("--idle-timeout", type=float, default=120.0)
    parser.add_argument("--finalize-timeout", type=float, default=300.0)
    parser.add_argument("--max-learning-attempts", type=int, default=1)
    parser.add_argument("--new-attempt", action="store_true")
    parser.add_argument("--note", help="run.py's --note (one line: why this run)")
    parser.add_argument("--force-resume", action="store_true")
    parser.add_argument("--keep-discarded-db", action="store_true")
    parser.add_argument("--stall-checkpoint", metavar="CID")
    parser.add_argument(
        "--stall-point", choices=("partial", "renamed"), default="partial"
    )
    parser.add_argument("--stall-flag", type=Path)
    parser.add_argument("--log-level", default="warning")
    return parser


def _stall(cid: str, point: str, flag: Path | None) -> None:
    """Block the capture of checkpoint ``cid`` for good (see the module docstring)."""

    def block() -> None:
        if flag is not None:
            flag.write_text(point)
        threading.Event().wait()  # in the capture's thread; the test kills the process

    if point == "partial":
        durable = run_module._write_json_durable

        def stalled(path: Path, value: Any) -> bytes:
            if path.name == CHECKPOINT_RECORD and path.parent.name == f"{cid}.partial":
                block()
            return durable(path, value)

        run_module._write_json_durable = stalled  # type: ignore[assignment]
        return
    atomic = run_module.write_json_atomic

    def stalled_latest(path: Path, value: Any, **kw: Any) -> None:
        if (
            path.name == "LATEST"
            and isinstance(value, Mapping)
            and value.get("id") == cid
        ):
            block()
        atomic(path, value, **kw)

    run_module.write_json_atomic = stalled_latest  # type: ignore[assignment]


def _pin_code(path: Path) -> None:
    """``protocol.code_fingerprint`` as the first invocation of the test saw it."""
    real = protocol.code_fingerprint

    def pinned(*, mirix: bool = False, **kw: Any) -> dict[str, Any]:
        key = ("with_mirix" if mirix else "harness_only") + (
            "_files" if kw.get("files") else ""
        )
        saved = json.loads(path.read_text()) if path.is_file() else {}
        if key not in saved:
            saved[key] = dict(real(mirix=mirix, **kw))
            write_json_atomic(path, saved)
        return saved[key]

    protocol.code_fingerprint = pinned  # type: ignore[assignment]


def _method_factory(name: str) -> Callable[[], Any] | None:
    if name == "storing":
        return lambda: StoringMethod([])
    if name == "mirix":
        from evals.skill_bench.methods.mirix import MirixMethod

        return lambda: MirixMethod(poll_s=0.1)
    return None  # the registered class (methods.base.method_class)


def _mirix_server(tag: str) -> Callable[[CellServerSpec, Any], MirixServer]:
    def factory(spec: CellServerSpec, mode: Any) -> MirixServer:
        return MirixServer(replace(spec, cell_id=f"{spec.cell_id}-{tag}"), mode=mode)

    return factory


def build_runner(args: argparse.Namespace) -> CellRunner:
    method = "reasoningbank" if args.method == "storing" else args.method
    spec = CellSpec("alfworld", "luna", method)
    options = RunOptions(
        out_dir=args.out,
        gateway_url=args.gateway_url,
        gateway_runs_dir=args.gateway_runs_dir,
        preflight=False,
        idle_timeout_s=args.idle_timeout,
        finalize_timeout_s=args.finalize_timeout,
        test_concurrency=1,
        max_learning_attempts=args.max_learning_attempts,
        new_attempt=args.new_attempt,
        note=args.note,
        force_resume=args.force_resume,
        keep_discarded_db=args.keep_discarded_db,
    )
    env = TinyEnv({"train": args.train, "val": args.val, "test": args.test})
    runner = CellRunner(
        spec,
        options,
        env_factory=lambda _benchmark: env,
        method_factory=_method_factory(args.method),
        git=lambda: dict(FIXED_GIT),
    )
    if args.db_tag:
        runner.mirix_server_factory = _mirix_server(args.db_tag)
    return runner


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=str(args.log_level).upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    if args.code_file is not None:
        _pin_code(args.code_file)
    if args.stall_checkpoint:
        _stall(args.stall_checkpoint, args.stall_point, args.stall_flag)
    if args.salt is not None:
        salt = CellPaths(args.out).salt
        if not salt.exists():
            salt.parent.mkdir(parents=True, exist_ok=True)
            salt.write_bytes(args.salt.encode("utf-8"))
    return run_cell(build_runner(args))


if __name__ == "__main__":
    raise SystemExit(main())
