"""Truncate a MetaClaw ``all_tests_metaclaw.json`` file to its first N days.

Pure function: reads the source path, writes the destination path, returns the
number of test entries kept.  Does not mutate the source file.  All top-level
keys are preserved verbatim; only the ``test`` array is truncated.

`n_days=0`  -> keep everything (full dataset).
`n_days>=N` -> keep everything (no upsizing).

The runner relies on this module for both arms so the dataset slice is
shared, which keeps the metaclaw-vs-mirix comparison input-identical.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union


def slice_tests(
    src_path: Union[str, Path], n_days: int, dst_path: Union[str, Path]
) -> int:
    """Read *src_path* JSON, truncate ``test[]`` to first *n_days*, write to *dst_path*.

    Args:
        src_path: Path to the source ``all_tests_metaclaw.json``.
        n_days: Number of days to keep (``0`` keeps all; values larger than the
            available test count keep all).
        dst_path: Output path for the truncated JSON.

    Returns:
        The number of test entries written to *dst_path*.
    """
    if n_days < 0:
        raise ValueError(f"n_days must be >= 0, got {n_days}")

    src = Path(src_path)
    dst = Path(dst_path)

    with src.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict) or "test" not in data:
        raise ValueError(
            f"{src} is not a MetaClaw test JSON (missing top-level 'test' key)"
        )

    tests = list(data.get("test") or [])
    if n_days == 0 or n_days >= len(tests):
        truncated = tests
    else:
        truncated = tests[:n_days]

    out = dict(data)
    out["test"] = truncated

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    return len(truncated)
