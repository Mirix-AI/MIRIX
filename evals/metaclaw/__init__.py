"""MetaClaw 30-day evaluation harness for MIRIX.

Vendors MetaClaw's runtime (skill backend + bench harness) and the 30-day
dataset under ``evals/metaclaw/{vendor,data}/``. The runtime has no dependency
on ``third_party/MetaClaw/``.

Supported arms are MetaClaw baselines plus ``--arm mirix-generic``. MIRIX runs
through the production memory path: add conversation turns with
``/memory/add_sync``, let MIRIX's automatic procedural trigger evolve memory,
and retrieve procedural memory through ``/memory/search``.
"""

from __future__ import annotations

from .dataset_slice import slice_tests  # noqa: F401
from .runner import RunResult, run_arm  # noqa: F401

__all__ = ["RunResult", "run_arm", "slice_tests"]
