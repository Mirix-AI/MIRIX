"""Tests for the test-only fault-injection registry.

The registry is the shared kernel that lets full-stack tests drive specific
SaveOutcome terminal states through the real worker. It must:

* Be a strict no-op when `settings.fault_injection_enabled` is False (the
  production default) — verified here so we can claim "zero prod impact".
* Resolve per-source directives from the `source_metadata.__fault_injection__`
  bag so concurrent FSTs keyed by unique source ids never collide.
* Map each fault shape to the right exception type so the EXISTING
  `error_policy.classify` produces the documented outcome with no classifier
  change.
* Honour `fail_attempts` so "transient that recovers on attempt N" works.
* Count fires per (source_key, site) so a passing test cannot mean the fault
  never fired.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mirix.errors import (
    CorrectableToolError,
    LLMChainingExhaustedError,
    ProviderConflictError,
    ProviderPermanentError,
    ProviderTransientError,
)
from mirix.testing import fault_injection as fi
from mirix.testing.fault_injection import SyntheticProviderError


@pytest.fixture(autouse=True)
def _clean_registry():
    """Each test starts with an empty registry and the flag forced on, except
    the explicit flag-off tests which patch it back."""
    fi.reset()
    with patch.object(fi.settings, "fault_injection_enabled", True):
        yield
    fi.reset()


# --------------------------------------------------------------------------- #
# No-op fast path — the prod-safety contract.
# --------------------------------------------------------------------------- #


def test_maybe_raise_is_noop_when_flag_off():
    """With the flag off, maybe_raise returns immediately and NEVER consults
    the registry — even if a directive was somehow registered."""
    with patch.object(fi.settings, "fault_injection_enabled", False):
        # A directive is present, but the flag gates everything.
        fi.resolve_directives(
            "src-1",
            {"__fault_injection__": {"faults": [{"site": "tool_body", "shape": "attribute_error"}]}},
        )
        # Must not raise, must not record a fire.
        fi.maybe_raise("tool_body", source_key="src-1")
    assert fi.fire_count("src-1", "tool_body") == 0


def test_resolve_directives_is_noop_when_flag_off():
    """resolve_directives does nothing when the flag is off."""
    with patch.object(fi.settings, "fault_injection_enabled", False):
        fi.resolve_directives(
            "src-x",
            {"__fault_injection__": {"faults": [{"site": "relational_write", "shape": "permanent"}]}},
        )
    # Re-enabling and asking should find nothing registered.
    with patch.object(fi.settings, "fault_injection_enabled", True):
        fi.maybe_raise("relational_write", source_key="src-x")  # no directive -> no raise


# --------------------------------------------------------------------------- #
# Shape -> exception mapping (drives error_policy classification).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "shape,exc_type",
    [
        ("attribute_error", AttributeError),
        ("transient", ProviderTransientError),
        ("permanent", ProviderPermanentError),
        ("conflict", ProviderConflictError),
        ("correctable", CorrectableToolError),
        ("llm_chaining_exhausted", LLMChainingExhaustedError),
        ("subagent_permanent", ProviderPermanentError),
        ("context_overflow", ValueError),
    ],
)
def test_shape_maps_to_exception(shape, exc_type):
    fi.resolve_directives(
        "src-shape",
        {"__fault_injection__": {"faults": [{"site": "tool_body", "shape": shape}]}},
    )
    with pytest.raises(exc_type):
        fi.maybe_raise("tool_body", source_key="src-shape")
    assert fi.fire_count("src-shape", "tool_body") == 1


def test_context_overflow_shape_is_recognized_as_overflow():
    """The context_overflow shape must raise an error that
    is_context_overflow_error() matches — otherwise inner_step's recovery branch
    never triggers and the FST would exercise nothing."""
    from mirix.llm_api.helpers import is_context_overflow_error

    fi.resolve_directives(
        "src-overflow",
        {"__fault_injection__": {"faults": [{"site": "llm_request", "shape": "context_overflow"}]}},
    )
    with pytest.raises(Exception) as excinfo:
        fi.maybe_raise("llm_request", source_key="src-overflow")
    assert is_context_overflow_error(excinfo.value), (
        "context_overflow fault must be recognized by is_context_overflow_error "
        "so the summarize-and-retry recovery path triggers"
    )


def test_unknown_shape_raises_valueerror_at_resolve():
    """A typo'd shape fails loudly at resolution, not silently at the hook."""
    with pytest.raises(ValueError):
        fi.resolve_directives(
            "src-bad",
            {"__fault_injection__": {"faults": [{"site": "tool_body", "shape": "nope"}]}},
        )


# --------------------------------------------------------------------------- #
# Site matching + per-source keying.
# --------------------------------------------------------------------------- #


def test_only_matching_site_fires():
    fi.resolve_directives(
        "src-2",
        {"__fault_injection__": {"faults": [{"site": "relational_write", "shape": "transient"}]}},
    )
    # Different site: no-op.
    fi.maybe_raise("tool_body", source_key="src-2")
    assert fi.fire_count("src-2", "tool_body") == 0
    # Matching provider site: fires as a status-bearing synthetic exception
    # (503 → the registered provider's translation maps it to transient).
    with pytest.raises(SyntheticProviderError) as ei:
        fi.maybe_raise("relational_write", source_key="src-2")
    assert ei.value.status_code == 503
    assert fi.fire_count("src-2", "relational_write") == 1


def test_directive_keyed_per_source():
    """A directive on src-A never fires for src-B (parallel-safety)."""
    fi.resolve_directives(
        "src-A",
        {"__fault_injection__": {"faults": [{"site": "relational_write", "shape": "permanent"}]}},
    )
    # src-B has no directive.
    fi.maybe_raise("relational_write", source_key="src-B")
    assert fi.fire_count("src-B", "relational_write") == 0
    with pytest.raises(SyntheticProviderError) as ei:
        fi.maybe_raise("relational_write", source_key="src-A")
    assert ei.value.status_code == 400


def test_tool_scoped_directive_only_fires_for_named_tool():
    """A directive scoped to a tool/memory_type fires only when ctx matches."""
    fi.resolve_directives(
        "src-tool",
        {
            "__fault_injection__": {
                "faults": [
                    {
                        "site": "subagent",
                        "shape": "subagent_permanent",
                        "tool": "episodic",
                    }
                ]
            }
        },
    )
    # Non-matching tool: no-op.
    fi.maybe_raise("subagent", source_key="src-tool", tool="semantic")
    assert fi.fire_count("src-tool", "subagent") == 0
    # Matching tool: fires.
    with pytest.raises(ProviderPermanentError):
        fi.maybe_raise("subagent", source_key="src-tool", tool="episodic")


# --------------------------------------------------------------------------- #
# fail_attempts: transient that recovers vs sustained.
# --------------------------------------------------------------------------- #


def test_fail_attempts_recovers_after_n():
    """fail_attempts=1 -> fires once then succeeds (recovers on attempt 2)."""
    fi.resolve_directives(
        "src-recover",
        {
            "__fault_injection__": {
                "faults": [
                    {
                        "site": "relational_write",
                        "shape": "transient",
                        "fail_attempts": 1,
                    }
                ]
            }
        },
    )
    with pytest.raises(SyntheticProviderError):
        fi.maybe_raise("relational_write", source_key="src-recover")  # attempt 1 -> fail
    fi.maybe_raise("relational_write", source_key="src-recover")  # attempt 2 -> recovered
    assert fi.fire_count("src-recover", "relational_write") == 1


def test_no_fail_attempts_means_sustained():
    """Omitting fail_attempts fires on every call (sustained failure)."""
    fi.resolve_directives(
        "src-sustained",
        {"__fault_injection__": {"faults": [{"site": "search_read", "shape": "transient"}]}},
    )
    for _ in range(3):
        with pytest.raises(SyntheticProviderError):
            fi.maybe_raise("search_read", source_key="src-sustained")
    assert fi.fire_count("src-sustained", "search_read") == 3


# --------------------------------------------------------------------------- #
# Fire logging (grepped off the aggregated service log by out-of-process FSTs).
# --------------------------------------------------------------------------- #


def test_fire_emits_prefixed_log_line(caplog):
    """Each fire logs a [FAULT INJECTION] line carrying site + source so the FST
    can grep the aggregated service log and count fires.
    """
    import logging

    fi.resolve_directives(
        "src-log",
        {"__fault_injection__": {"faults": [{"site": "relational_write", "shape": "permanent"}]}},
    )
    # The "Mirix" logger sets propagate=False, so caplog (which captures via the
    # root logger) needs propagation re-enabled for the duration of the assert.
    mirix_logger = logging.getLogger("Mirix")
    prior_propagate = mirix_logger.propagate
    mirix_logger.propagate = True
    try:
        with caplog.at_level("WARNING", logger="Mirix"):
            with pytest.raises(SyntheticProviderError):
                fi.maybe_raise("relational_write", source_key="src-log")
    finally:
        mirix_logger.propagate = prior_propagate

    fire_records = [r for r in caplog.records if fi.LOG_PREFIX in r.getMessage() and "fired" in r.getMessage()]
    assert len(fire_records) == 1
    # Pin the logger tree: the fire line must ride the "Mirix" handler stack.
    assert fire_records[0].name == "Mirix"
    fire_line = fire_records[0].getMessage()
    assert "site=relational_write" in fire_line
    assert "source=src-log" in fire_line
    assert "shape=permanent" in fire_line


def test_next_fault_returns_shape_without_raising():
    """next_fault (used by a search-read hook) records the fire and returns the
    matched shape so the caller can raise its own native (httpx) exception."""
    fi.resolve_directives(
        "src-next",
        {"__fault_injection__": {"faults": [{"site": "search_read", "shape": "transient"}]}},
    )
    shape = fi.next_fault("search_read", source_key="src-next")
    assert shape == "transient"
    assert fi.fire_count("src-next", "search_read") == 1
    # No directive for a different source -> None, no fire.
    assert fi.next_fault("search_read", source_key="src-none") is None


def test_next_fault_is_noop_when_off():
    with patch.object(fi.settings, "fault_injection_enabled", True):
        fi.resolve_directives(
            "src-no",
            {"__fault_injection__": {"faults": [{"site": "search_read", "shape": "transient"}]}},
        )
    with patch.object(fi.settings, "fault_injection_enabled", False):
        assert fi.next_fault("search_read", source_key="src-no") is None
    assert fi.fire_count("src-no", "search_read") == 0


# --------------------------------------------------------------------------- #
# Prod-env gate — the second, independent safety check (belt and suspenders).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("prod_env", ["prod", "PROD", "production", "prd"])
def test_prod_env_hard_disables_even_when_flag_on(prod_env, monkeypatch):
    """Even with the flag forced True, a production APP_ENV keeps the seam inert.
    The autouse fixture already forces the flag on."""
    monkeypatch.setenv("APP_ENV", prod_env)
    fi.resolve_directives(
        "src-prod",
        {"__fault_injection__": {"faults": [{"site": "tool_body", "shape": "attribute_error"}]}},
    )
    # resolve was a no-op under prod, so nothing fires.
    fi.maybe_raise("tool_body", source_key="src-prod")
    assert fi.fire_count("src-prod", "tool_body") == 0


def test_non_prod_env_allows_injection(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    fi.resolve_directives(
        "src-test-env",
        {"__fault_injection__": {"faults": [{"site": "tool_body", "shape": "attribute_error"}]}},
    )
    with pytest.raises(AttributeError):
        fi.maybe_raise("tool_body", source_key="src-test-env")
