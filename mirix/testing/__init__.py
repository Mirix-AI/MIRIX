"""Test-only support code.

Modules in this package are imported by prod code paths but are inert unless a
test-only setting (e.g. ``settings.fault_injection_enabled``) is turned on and
the process is not in a production app environment. Nothing here runs in
production with default settings.
"""
