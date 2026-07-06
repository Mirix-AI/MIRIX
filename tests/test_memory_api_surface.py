"""Production REST surface for memory APIs.

Procedural memory is learned by the automatic conversation-store distillation
pipeline. The public REST surface should expose generic memory ingestion and
search, not direct skill lifecycle controls.
"""

from mirix.server.rest_api import router
from mirix.server.rest_api import _procedural_memory_response


def _paths() -> set[str]:
    return {
        route.path
        for route in router.routes
        if hasattr(route, "path") and getattr(route, "include_in_schema", True)
    }


def test_public_memory_surface_excludes_direct_skill_routes():
    paths = _paths()

    assert "/memory/add" in paths
    assert "/memory/add_sync" in paths
    assert "/memory/search" in paths
    assert "/memory/search_all_users" in paths

    assert not any(path.startswith("/v1/skills") for path in paths)
    assert not any(path.startswith("/memory/procedural") for path in paths)


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
