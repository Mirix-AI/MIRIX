"""
Tests for mirix.database.provider_validation.

Usage:
    pytest tests/test_provider_validation.py -v
"""

import pytest

from mirix.database.provider_validation import validate_provider_pairing_or_raise
from mirix.database.relational_provider import (
    get_registered_relational_providers,
    register_relational_provider,
    reset_provider_mode_latch,
    unregister_relational_provider,
)
from mirix.database.search_provider import (
    get_registered_search_providers,
    register_search_provider,
    unregister_search_provider,
)


@pytest.fixture(autouse=True)
def cleanup_registries():
    for name in list(get_registered_relational_providers().keys()):
        unregister_relational_provider(name)
    for name in list(get_registered_search_providers().keys()):
        unregister_search_provider(name)
    reset_provider_mode_latch()
    yield
    for name in list(get_registered_relational_providers().keys()):
        unregister_relational_provider(name)
    for name in list(get_registered_search_providers().keys()):
        unregister_search_provider(name)
    reset_provider_mode_latch()


class TestValidateProviderPairing:
    def test_passes_when_both_registered(self):
        register_relational_provider("r", object())
        register_search_provider("s", object())
        validate_provider_pairing_or_raise()

    def test_passes_when_neither_registered(self):
        validate_provider_pairing_or_raise()

    def test_raises_when_only_relational_registered(self):
        register_relational_provider("r", object())
        with pytest.raises(RuntimeError, match="Relational DB provider is registered"):
            validate_provider_pairing_or_raise()

    def test_raises_when_only_search_registered(self):
        register_search_provider("s", object())
        with pytest.raises(RuntimeError, match="Search provider is registered"):
            validate_provider_pairing_or_raise()
