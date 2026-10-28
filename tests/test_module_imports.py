"""Every ``mirix`` module imports without provider keys.

A fresh interpreter walks the whole package with ``pkgutil.walk_packages`` and
imports each module, with every model-provider key unset, ``HOME`` pointed at
an empty directory (so ``~/.mirix`` is never read) and no ``MIRIX_PG_URI``. A
module that only fails at import time — a dangling import of a package that no
longer exists, a key read at import — is invisible to tests that never import
it; this catches it.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PROVIDER_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
    "AZURE_API_KEY",
)

WALK = """
import importlib, json, pkgutil
import mirix
failed = {}
names = []
def on_error(name):
    failed[name] = "walk error"
for info in pkgutil.walk_packages(mirix.__path__, "mirix.", onerror=on_error):
    names.append(info.name)
    try:
        importlib.import_module(info.name)
    except BaseException as exc:
        failed[info.name] = f"{type(exc).__name__}: {exc}"
print(json.dumps({"count": len(names), "failed": failed}))
"""


def test_every_module_imports_without_keys(tmp_path):
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in PROVIDER_KEYS and key != "MIRIX_PG_URI"
    }
    env["HOME"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", WALK],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["failed"] == {}
    assert report["count"] > 200  # the walk really covered the package
