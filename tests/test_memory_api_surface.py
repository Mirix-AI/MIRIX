"""Production REST surface for memory APIs.

Procedural memory is learned by the automatic conversation-store distillation
pipeline. The public REST surface exposes generic memory ingestion and search,
plus DELETE for parity with every other memory type — but no direct skill
WRITE controls (create/edit go through the evolution flow only).
"""

from mirix.server.rest_api import router
from mirix.server.rest_api import _procedural_memory_response


def _routes() -> list:
    return [
        route
        for route in router.routes
        if hasattr(route, "path") and getattr(route, "include_in_schema", True)
    ]


def _paths() -> set[str]:
    return {route.path for route in _routes()}


def test_public_memory_surface_excludes_direct_skill_write_routes():
    paths = _paths()

    assert "/memory/add" in paths
    assert "/memory/add_sync" in paths
    assert "/memory/search" in paths
    assert "/memory/search_all_users" in paths

    assert not any(path.startswith("/v1/skills") for path in paths)

    # DELETE stays (uniform with episodic/semantic/resource/knowledge_vault);
    # any other method on /memory/procedural* is a skill write and must not
    # be public.
    procedural_methods = {
        method
        for route in _routes()
        if route.path.startswith("/memory/procedural")
        for method in (getattr(route, "methods", None) or set())
    }
    assert procedural_methods == {"DELETE"}


def test_procedural_delete_route_is_uniform_with_other_memory_types():
    paths = _paths()
    for memory_type in ("episodic", "semantic", "procedural", "resource"):
        assert f"/memory/{memory_type}/{{memory_id}}" in paths


def test_procedural_read_shape_is_full_skill_schema():
    from datetime import datetime
    from types import SimpleNamespace

    item = SimpleNamespace(
        id="proc-1",
        entry_type="workflow",
        name="deploy-production",
        description="Deploy the service",
        instructions="Run tests, merge, and monitor.",
        triggers=["user asks to deploy"],
        examples=[{"input": "deploy", "output": "checklist"}],
        version="0.2.0",
        created_at=datetime(2026, 1, 1, 12, 0, 0),
        updated_at=datetime(2026, 1, 2, 12, 0, 0),
        user_id="user-1",
    )

    row = _procedural_memory_response(item, include_user_id=True)

    assert row == {
        "memory_type": "procedural",
        "id": "proc-1",
        "entry_type": "workflow",
        "name": "deploy-production",
        "description": "Deploy the service",
        "instructions": "Run tests, merge, and monitor.",
        "triggers": ["user asks to deploy"],
        "examples": [{"input": "deploy", "output": "checklist"}],
        "version": "0.2.0",
        "created_at": "2026-01-01T12:00:00",
        "updated_at": "2026-01-02T12:00:00",
        "user_id": "user-1",
    }
