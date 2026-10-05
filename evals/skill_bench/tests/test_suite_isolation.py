"""The harness conftest isolates the harness tests from another suite in the same pytest
process, both ways (``pytest tests/ evals/skill_bench/tests``, the default testpaths).

A child pytest runs three directories in order: ``mirix_like`` (a conftest that turns the
tokenizer vocabulary off for the whole session, as MIRIX's ``tests/conftest.py`` does),
``harness`` (this directory's conftest, copied) and ``mirix_after`` (under the first
directory's session fixture, as the rest of MIRIX's tests would be). The child's
environment carries a made-up provider key. The harness module must be imported, and
its test run, without the key and with the real o200k encoding; the other suite's
modules and tests, before and after it, must see the key and their own tokenizer switch.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from evals.skill_bench import protocol

HERE = Path(__file__).parent
FAKE_KEY = "not-a-real-key-suite-isolation"
KEYS = ("OPENROUTER_API_KEY", "GEMINI_API_KEY")

MIRIX_LIKE_CONFTEST = """
import pytest


@pytest.fixture(scope="session", autouse=True)
def _no_tokenizer_download(tmp_path_factory):
    import tiktoken.load

    def no_download(path, *args, **kwargs):
        raise RuntimeError("the vocabulary is switched off in this suite")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(tiktoken.load, "read_file", no_download)
        patch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path_factory.mktemp("tiktoken")))
        yield
"""

OTHER_TEST = """
import os

IMPORTED_WITH = os.environ.get("GEMINI_API_KEY")


def test_the_other_suite_keeps_its_environment():
    assert IMPORTED_WITH == {key!r}  # also when collected after the harness conftest
    assert os.environ.get("GEMINI_API_KEY") == {key!r}
    assert os.path.basename(os.environ["TIKTOKEN_CACHE_DIR"]) == "tiktoken0"
    import tiktoken.load

    assert tiktoken.load.read_file.__name__ == "no_download"
"""

HARNESS_TEST = """
import os

import pytest

from evals.skill_bench.infra import tokens

IMPORTED_WITH = [n for n in {keys!r} if os.environ.get(n)]


@pytest.fixture(scope="module")
def keys_at_module_setup():
    # Set up before any function-scoped fixture (the conftest's autouse one included).
    return [n for n in {keys!r} if os.environ.get(n)]


def test_the_harness_sees_no_key_and_the_real_tokenizer(keys_at_module_setup):
    assert IMPORTED_WITH == []  # nor did its module when it was imported
    assert keys_at_module_setup == []  # nor its fixtures of a wider scope
    assert not any(os.environ.get(n) for n in {keys!r})
    tokens.count_text("hello world")
    assert tokens.encoding_name() == {encoding!r}
"""


def test_harness_and_other_suite_in_one_process(tmp_path):
    import tiktoken

    try:  # the real vocabulary must be loadable here (it is cached with the harness)
        tiktoken.get_encoding(protocol.TOKEN_ENCODING)
    except Exception as exc:  # pragma: no cover - no cached vocabulary on this machine
        pytest.skip(f"o200k vocabulary unavailable: {exc}")
    for name, conftest, test in (
        ("mirix_like", MIRIX_LIKE_CONFTEST, OTHER_TEST),
        ("harness", (HERE / "conftest.py").read_text(), HARNESS_TEST),
        # No conftest of its own: it runs under the first suite's session fixture, which
        # the harness tests suspended and must have put back.
        ("mirix_after", None, OTHER_TEST),
    ):
        directory = tmp_path / name
        directory.mkdir()
        if conftest is not None:
            (directory / "conftest.py").write_text(conftest)
        (directory / f"test_{name}.py").write_text(
            test.format(key=FAKE_KEY, keys=KEYS, encoding=protocol.TOKEN_ENCODING)
        )
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
    }
    env.update(
        GEMINI_API_KEY=FAKE_KEY,
        PYTHONPATH=str(protocol.REPO_ROOT),
        HOME=str(tmp_path / "home"),
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            "--import-mode=importlib",
            "-o",
            "addopts=",
            "mirix_like",
            "harness",
            "mirix_after",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
    assert "3 passed" in result.stdout


NEEDS_VOCABULARY_TEST = """
from evals.skill_bench.infra import tokens


def test_counts_with_the_real_vocabulary():
    tokens.reset_encoding_for_tests()
    tokens.count_text("hello world")
    assert tokens.encoding_name() == {encoding!r}
"""


def _child_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
        and k not in ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    }
    env.update(PYTHONPATH=str(protocol.REPO_ROOT), HOME=str(tmp_path / "home"), **extra)
    return env


def test_a_harness_test_never_downloads_the_vocabulary(tmp_path):
    """With the o200k vocabulary missing from tiktoken's cache, a harness test
    is skipped with the command that caches it; nothing is fetched (a download would
    fill the cache directory; every proxy points at a closed loopback port as well)."""
    cache = tmp_path / "empty-tiktoken-cache"
    cache.mkdir()
    directory = tmp_path / "harness"
    directory.mkdir()
    (directory / "conftest.py").write_text((HERE / "conftest.py").read_text())
    (directory / "test_needs_vocabulary.py").write_text(
        NEEDS_VOCABULARY_TEST.format(encoding=protocol.TOKEN_ENCODING)
    )
    dead = "http://127.0.0.1:9"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            "-rs",
            "--import-mode=importlib",
            "-o",
            "addopts=",
            "harness",
        ],
        cwd=tmp_path,
        env=_child_env(
            tmp_path,
            TIKTOKEN_CACHE_DIR=str(cache),
            HTTPS_PROXY=dead,
            HTTP_PROXY=dead,
            ALL_PROXY=dead,
        ),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
    assert "1 skipped" in result.stdout, result.stdout[-4000:]
    assert "tests never download it" in result.stdout
    assert "tiktoken.get_encoding('o200k_base')" in result.stdout
    assert list(cache.iterdir()) == []


#: The optional ``[skill-bench]`` extra (pyproject.toml) that a plain MIRIX install lacks.
SKILL_BENCH_ONLY_PACKAGES = (
    "alfworld",
    "textworld",
)

COLLECT_WITHOUT_EXTRAS = """
import importlib.abc
import sys

HIDDEN = {hidden!r}


class Hide(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in HIDDEN:
            raise ModuleNotFoundError(f"No module named {{name!r}} (hidden)", name=name)
        return None


sys.meta_path.insert(0, Hide())
import pytest

sys.exit(
    pytest.main(["--collect-only", "-q", "-p", "no:cacheprovider", {directory!r}])
)
"""


def test_the_harness_suite_collects_without_the_skill_bench_extras(tmp_path):
    """The default testpaths include this directory, so on a plain MIRIX
    install (``pip install -e .`` without ``[skill-bench]``) a bare ``pytest`` must not
    abort at collection: every harness module imports, and what needs an extra skips."""
    runner = tmp_path / "collect_without_extras.py"
    runner.write_text(
        COLLECT_WITHOUT_EXTRAS.format(
            hidden=SKILL_BENCH_ONLY_PACKAGES, directory=str(HERE)
        )
    )
    result = subprocess.run(
        [sys.executable, str(runner)],
        cwd=protocol.REPO_ROOT,
        env=_child_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=600,
    )
    tail = result.stdout[-4000:] + result.stderr[-2000:]
    assert result.returncode == 0, tail
    summary = result.stdout.strip().splitlines()[-1]
    assert "error" not in summary.lower() and "collected" in summary, tail


def test_the_tests_read_a_copy_of_the_frozen_record():
    """run.py freezes automatically, so during the tests every reader and writer not
    given a path uses the session's copy of the record, frozen to this tree, never the
    repository's ``config/protocol_frozen.json``."""
    path = protocol.frozen_record_path()
    assert path != protocol.FROZEN_PROTOCOL_JSON
    assert os.environ[protocol.FROZEN_RECORD_ENV] == str(path) and path.is_file()
    committed = protocol.load_frozen_protocol(protocol.FROZEN_PROTOCOL_JSON)
    known = protocol.known_versions()
    assert committed["fingerprint"] in known
    assert {e["fingerprint"] for e in committed["history"]} <= set(known)
