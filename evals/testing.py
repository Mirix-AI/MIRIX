"""Shared helpers for the eval harnesses' server wire-contract tests.

Both `evals/metaclaw/tests/test_server_contract.py` and
`evals/alfworld/tests/test_server_contract.py` introspect the real MIRIX
FastAPI router; the introspection lives here so the two suites cannot drift
apart. Test-only — nothing in the harness runtime paths imports this module.
"""

from __future__ import annotations

import inspect

from mirix.server import rest_api
from mirix.server.rest_api import router


def routes_by_path() -> dict[str, set[str]]:
    """Map every router path to the set of HTTP methods it serves."""
    routes: dict[str, set[str]] = {}
    for route in router.routes:
        if hasattr(route, "path"):
            routes.setdefault(route.path, set()).update(
                getattr(route, "methods", None) or set()
            )
    return routes


def query_param_names(path: str, method: str) -> set[str]:
    """Return the declared query-parameter names for one path+method route."""
    names: set[str] = set()
    for route in router.routes:
        if getattr(route, "path", None) != path:
            continue
        if method not in (getattr(route, "methods", None) or set()):
            continue
        dependant = getattr(route, "dependant", None)
        if dependant is not None:
            names.update(param.name for param in dependant.query_params)
    return names


def endpoint_source(handler_name: str) -> str:
    """Source of a rest_api endpoint handler, unwrapped past decorators.

    Some response envelopes are handler-built plain dicts with no response
    model to pin (and this branch keeps mirix/ byte-identical to the
    production PR, so none can be added here); a source-level pin still
    catches an envelope key being renamed or dropped.
    """
    handler = getattr(rest_api, handler_name)
    return inspect.getsource(inspect.unwrap(handler))
