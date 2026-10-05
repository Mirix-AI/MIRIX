"""skill_bench test isolation: no provider keys, loopback without proxies, and the real
tokenizer even when MIRIX's own tests run in the same pytest process.

``pytest -m "not integration" tests/ evals/skill_bench/tests`` runs MIRIX's unit tests and
these in one process. MIRIX's ``tests/conftest.py`` turns the o200k vocabulary off for its
whole session (a session-scoped autouse fixture: ``tiktoken.load.read_file`` raises and
``TIKTOKEN_CACHE_DIR`` points at an empty directory). A session fixture stays active until
the session ends, so every harness test after the first MIRIX test would count with the
chars/3 fallback: the gateway is then not protocol-conformant (H3) and every cell is
refused. The ``pytest_runtest_setup`` / ``pytest_runtest_teardown`` wrappers below (they
apply to the tests of this directory only) put back, around each harness test and its
fixtures of every scope, what the process had when this conftest was imported (at
collection, before any test ran), and restore the other suite's state afterwards.

The vocabulary is never downloaded by a test: the ``read_file`` put back refuses an
``http(s)`` location, so it only reads tiktoken's cache (``TIKTOKEN_CACHE_DIR``, else
tiktoken's temp-directory cache). When the o200k vocabulary is not cached there, every
harness test is skipped with the one command that caches it (:data:`VOCABULARY_MISSING`),
instead of fetching it from the network.

Provider keys are kept away the same way, and only from the harness: no harness module is
imported (``pytest_make_collect_report`` of this directory's collectors), and no harness
test or fixture runs, with a provider key in the environment, while the other suite's
modules and tests, collected and run before or after, keep the environment they were
given (a MIRIX test that needs ``GEMINI_API_KEY`` is not silently skipped because the
harness tests share its process).

The frozen protocol record is a copy (:func:`_frozen_record_copy`): ``run.py`` freezes the
tree automatically when an attempt starts or resumes under a tree that is not the frozen
value, so a cell a test runs on an unfrozen checkout would write the
repository's ``config/protocol_frozen.json``. For the whole session every reader and writer
not given a path (``protocol.frozen_record_path``, cell subprocesses included: they inherit
the environment) uses a copy frozen to the tree, and the session fails if the repository's
record changed anyway. A test that freezes to another tree points
``SKILL_BENCH_FROZEN_RECORD`` at its own copy (``monkeypatch.setenv``).
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

# Safety net on top of the documented `env -u ...` invocation: no harness test may reach a
# real provider, so provider keys are out of the environment while a harness module is
# imported and while a harness test runs (the hooks below).
PROVIDER_KEYS = (
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)
#: The skip reason of every harness test when the tokenizer vocabulary is not cached.
VOCABULARY_MISSING = (
    "the o200k_base tokenizer vocabulary is not in tiktoken's cache and the tests never "
    'download it; cache it once with: python -c "import tiktoken; '
    "tiktoken.get_encoding('o200k_base')\""
)
_loopback = "127.0.0.1,localhost,::1"
os.environ["NO_PROXY"] = ",".join(
    filter(None, [os.environ.get("NO_PROXY", ""), _loopback])
)
os.environ["no_proxy"] = os.environ["NO_PROXY"]


def _cache_only(read_file: Any) -> Any:
    """``tiktoken.load.read_file`` that never downloads: an ``http(s)`` location (what
    tiktoken reads when the vocabulary is not in its cache) raises instead."""

    def read_local_file(blobpath: str, *args: Any, **kwargs: Any) -> Any:
        if str(blobpath).lower().startswith(("http://", "https://")):
            raise OSError(f"harness tests never download {blobpath}")
        return read_file(blobpath, *args, **kwargs)

    read_local_file.__wrapped__ = read_file  # type: ignore[attr-defined]
    return read_local_file


class _Tokenizer:
    """The tokenizer state of the process: ``tiktoken.load.read_file`` (wrapped so that it
    never downloads) and ``TIKTOKEN_CACHE_DIR``, as captured at collection, and what
    another suite had set while a harness test runs."""

    def __init__(self) -> None:
        import tiktoken.load

        self.module: Any = tiktoken.load
        self.read_file = _cache_only(tiktoken.load.read_file)
        self.cache_dir = os.environ.get("TIKTOKEN_CACHE_DIR")
        self.saved: tuple[Any, str | None] | None = None
        self._available: bool | None = None

    def available(self) -> bool:
        """Whether the o200k vocabulary loads from the cache (checked once, installed)."""
        if self._available is None:
            import tiktoken

            from evals.skill_bench import protocol

            try:
                tiktoken.get_encoding(protocol.TOKEN_ENCODING)
            except Exception:  # not cached (and never downloaded), or unreadable
                self._available = False
            else:
                self._available = True
        return self._available

    @staticmethod
    def _set_cache_dir(value: str | None) -> None:
        if value is None:
            os.environ.pop("TIKTOKEN_CACHE_DIR", None)
        else:
            os.environ["TIKTOKEN_CACHE_DIR"] = value

    def install(self) -> None:
        self.saved = (self.module.read_file, os.environ.get("TIKTOKEN_CACHE_DIR"))
        self.module.read_file = self.read_file
        self._set_cache_dir(self.cache_dir)
        from evals.skill_bench.infra import tokens

        if tokens.encoding_name() == tokens.FALLBACK_ENCODING:
            # A fallback remembered while the vocabulary was switched off (by the other
            # suite) is forgotten: the next count loads the real encoding.
            tokens.reset_encoding_for_tests()

    def restore(self) -> None:
        if self.saved is None:
            return
        read_file, cache_dir = self.saved
        self.saved = None
        self.module.read_file = read_file
        self._set_cache_dir(cache_dir)


_TOKENIZER = _Tokenizer()


class _ProviderKeys:
    """Provider keys out of the environment while a harness module is collected or a
    harness test (and its fixtures of every scope) runs; whatever was there before is put
    back afterwards."""

    def __init__(self) -> None:
        self.saved: list[dict[str, str]] = []  # a stack: a nested removal is harmless

    def remove(self) -> None:
        self.saved.append(
            {n: os.environ.pop(n) for n in PROVIDER_KEYS if n in os.environ}
        )

    def restore(self) -> None:
        if not self.saved:
            return
        saved = self.saved.pop()
        for (
            name
        ) in PROVIDER_KEYS:  # a test may have set one: it never outlives the test
            os.environ.pop(name, None)
        os.environ.update(saved)


_KEYS = _ProviderKeys()


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector: pytest.Collector) -> Iterator[None]:
    # Collecting a module of this directory imports it (and what it imports).
    _KEYS.remove()
    try:
        yield
    finally:
        _KEYS.restore()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item: pytest.Item) -> Iterator[None]:
    # Before the item's fixtures of any scope are set up.
    _KEYS.remove()
    _TOKENIZER.install()
    if not _TOKENIZER.available():
        pytest.skip(VOCABULARY_MISSING)  # the teardown hook below still restores
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(
    item: pytest.Item, nextitem: pytest.Item | None
) -> Iterator[None]:
    yield  # the item's fixtures (and a module's or the session's, if last) are torn down
    _TOKENIZER.restore()
    _KEYS.restore()


@pytest.fixture(scope="session", autouse=True)
def _frozen_record_copy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """The session's frozen record: a copy of the repository's, frozen to this tree
    (a no-op on a frozen tree, so a dirty checkout never makes the tests freeze the real
    record), named by ``SKILL_BENCH_FROZEN_RECORD`` until the session ends. The
    repository's record must be byte-identical afterwards."""
    from evals.skill_bench import protocol
    from evals.skill_bench.tools import freeze_protocol

    committed = protocol.FROZEN_PROTOCOL_JSON.read_bytes()
    copy = tmp_path_factory.mktemp("frozen_record") / "protocol_frozen.json"
    shutil.copyfile(protocol.FROZEN_PROTOCOL_JSON, copy)
    freeze_protocol.freeze(reason="the tests baseline", path=copy)
    previous = os.environ.get(protocol.FROZEN_RECORD_ENV)
    os.environ[protocol.FROZEN_RECORD_ENV] = str(copy)
    try:
        yield copy
    finally:
        if previous is None:
            os.environ.pop(protocol.FROZEN_RECORD_ENV, None)
        else:
            os.environ[protocol.FROZEN_RECORD_ENV] = previous
    if protocol.FROZEN_PROTOCOL_JSON.read_bytes() != committed:
        raise AssertionError(
            f"a test wrote the repository's frozen record {protocol.FROZEN_PROTOCOL_JSON}"
            " (restore it from git): point SKILL_BENCH_FROZEN_RECORD at a copy"
        )


@pytest.fixture(autouse=True)
def _no_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in PROVIDER_KEYS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session", autouse=True)
def _session_host_dir(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """The session's host directory (``host_registry.py``), for anything a session or
    module fixture starts before :func:`_isolated_host_dir` gives each test its own."""
    from evals.skill_bench import host_registry

    directory = tmp_path_factory.mktemp("host_session")
    previous = os.environ.get(host_registry.HOST_DIR_ENV)
    os.environ[host_registry.HOST_DIR_ENV] = str(directory)
    try:
        yield directory
    finally:
        if previous is None:
            os.environ.pop(host_registry.HOST_DIR_ENV, None)
        else:
            os.environ[host_registry.HOST_DIR_ENV] = previous


@pytest.fixture(autouse=True)
def _isolated_host_dir(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Each test's schedulers coordinate in a host directory of their own
    (``host_registry.py``; subprocesses inherit it): never the real
    ``~/.cache/skill_bench/host`` a scheduler of this machine uses, never another
    test's."""
    from evals.skill_bench import host_registry

    directory = tmp_path_factory.mktemp("host")
    monkeypatch.setenv(host_registry.HOST_DIR_ENV, str(directory))
    return directory


@pytest.fixture(scope="session")
def e2e_matrix(tmp_path_factory: pytest.TempPathFactory):
    """The offline end-to-end matrix (``tests/e2e_support.py``), built once per
    session and shared read-only by ``test_e2e.py`` and ``test_audit.py`` (tests that tamper
    with records work on copies)."""
    from evals.skill_bench.tests.e2e_support import build_matrix

    return build_matrix(tmp_path_factory.mktemp("e2e"))


#: The method the removal tests record as removed (:func:`removed_method`): a name that
#: fits the cell-id grammar and is no protocol method.
REMOVED_METHOD = "old_baseline"


@pytest.fixture
def removed_method(monkeypatch: pytest.MonkeyPatch) -> str:
    """Record :data:`REMOVED_METHOD` as removed from the study for one test and return its
    name. ``study.REMOVED_METHODS`` names no method in this release, so the tests of the
    removal mechanism (refusals, set-aside cells, listings) install one of their own."""
    from types import MappingProxyType

    from evals.skill_bench import study

    monkeypatch.setattr(
        study,
        "REMOVED_METHODS",
        MappingProxyType(
            {
                REMOVED_METHOD: study.Removal(
                    decision="example removal",
                    reason="An example reason: the method left the study.",
                )
            }
        ),
    )
    return REMOVED_METHOD
