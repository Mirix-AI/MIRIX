"""MetaClaw 30-day evaluation harness for MIRIX.

Vendors MetaClaw's runtime (skill backend + bench harness) and the 30-day
dataset under ``evals/metaclaw/{vendor,data}/``.  The runtime has no
dependency on ``third_party/MetaClaw/``.

Slice #1 (this code): only ``--arm metaclaw`` is implemented end-to-end;
MIRIX-as-skill-backend arms land in subsequent slices.
"""

from __future__ import annotations

from .dataset_slice import slice_tests  # noqa: F401
from .runner import RunResult, run_arm  # noqa: F401

__all__ = ["RunResult", "run_arm", "slice_tests"]
