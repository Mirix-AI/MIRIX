"""Run SkillOpt 79124b3's own code from its read-only clone, for byte-level parity tests.

SkillOpt's package ``__init__`` files import heavy optional dependencies (model
backends, gymnasium, ALFWorld workers). :func:`skillopt_modules` replaces those
package modules with empty namespace stubs whose ``__path__`` points into the clone, so
``importlib.import_module("skillopt.envs.alfworld.vendor.env_manager")`` loads the real
file and nothing else. ``skillopt.model`` becomes a stub carrying the functions the test
passes in (for example a recording ``chat_target``). Everything is removed from
``sys.modules`` again on exit.

Set ``SKILLOPT_DIR`` to a SkillOpt 79124b3 clone; the tests that need it skip when it is
not set or the clone is missing.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

#: The SkillOpt clone, from the environment only (None: not set).
SKILLOPT_DIR: Path | None = (
    Path(os.environ["SKILLOPT_DIR"]) if os.environ.get("SKILLOPT_DIR") else None
)

# Packages whose __init__ would import heavy dependencies: stubbed as namespaces.
_NAMESPACE_PACKAGES = {
    "skillopt": "skillopt",
    "skillopt.envs": "skillopt/envs",
    "skillopt.envs.alfworld": "skillopt/envs/alfworld",
    "skillopt.envs.alfworld.vendor": "skillopt/envs/alfworld/vendor",
}


def skillopt_available() -> bool:
    return (
        SKILLOPT_DIR is not None
        and (SKILLOPT_DIR / "skillopt" / "envs" / "alfworld" / "rollout.py").is_file()
    )


requires_skillopt = pytest.mark.skipif(
    not skillopt_available(),
    reason="SkillOpt clone not found (set SKILLOPT_DIR)"
    if SKILLOPT_DIR is None
    else f"SkillOpt clone not found at {SKILLOPT_DIR}",
)


def _skillopt_dir() -> Path:
    if SKILLOPT_DIR is None:
        pytest.skip("SKILLOPT_DIR is not set")
    return SKILLOPT_DIR


def skillopt_path(rel: str) -> Path:
    return _skillopt_dir() / rel


def _unavailable(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("this SkillOpt backend is not available in tests")


def _is_skillopt(name: str) -> bool:
    return name == "skillopt" or name.startswith("skillopt.")


@contextlib.contextmanager
def skillopt_modules(
    **model_functions: Callable[..., Any],
) -> Iterator[Callable[[str], Any]]:
    """Yield ``import_module`` with SkillOpt importable from the clone (see module docstring)."""
    root = _skillopt_dir()
    saved = {name: mod for name, mod in sys.modules.items() if _is_skillopt(name)}
    for name in saved:
        del sys.modules[name]
    try:
        for name, rel in _NAMESPACE_PACKAGES.items():
            package = types.ModuleType(name)
            package.__path__ = [str(root / rel)]
            sys.modules[name] = package
        model = types.ModuleType("skillopt.model")
        model.__path__ = []
        defaults = {
            "chat_target": _unavailable,
            "chat_target_messages": _unavailable,
            "get_target_backend": lambda: "openai_compatible",
            "is_target_exec_backend": lambda: False,
        }
        for attr, fn in {**defaults, **model_functions}.items():
            setattr(model, attr, fn)
        sys.modules["skillopt.model"] = model
        harness = types.ModuleType("skillopt.model.codex_harness")
        for attr in ("prepare_workspace", "render_skill_md", "run_target_exec"):
            setattr(harness, attr, _unavailable)
        sys.modules["skillopt.model.codex_harness"] = harness
        yield importlib.import_module
    finally:
        for name in [n for n in sys.modules if _is_skillopt(n)]:
            del sys.modules[name]
        sys.modules.update(saved)
