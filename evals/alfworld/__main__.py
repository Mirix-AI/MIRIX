"""Run the ALFWorld eval harness with ``python -m evals.alfworld``."""

from __future__ import annotations

import sys

from .env import ALFWorldDependencyError
from .runner import main


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ALFWorldDependencyError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
