"""Static wire-contract pins between the ALFWorld harness and the MIRIX server.

The harness is pure-HTTP (zero ``mirix`` imports in the runtime path), so these
tests are the only import-time coupling: they load the REAL server router and
Pydantic models and pin every request/response feature the harness depends on.
An incompatible server-side API change fails here, not mid-benchmark.

Pinned against ``evals/alfworld/mirix_adapter.py`` and ``runner.py``.
"""

import pytest
from pydantic import ValidationError

from evals.alfworld.mirix_adapter import (
    build_add_sync_payload,
    build_consolidation_session_id,
    build_episode_session_id,
    build_meta_agent_config,
)
from mirix.schemas.agent import AgentState
from mirix.schemas.auto_dream import AutoDreamRequest, AutoDreamResponse
from mirix.schemas.client import Client, ClientUpdate
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.procedural_memory import ProceduralMemoryItem
from mirix.server.rest_api import (
    AddMemoryRequest,
    CreateOrGetClientRequest,
    InitializeMetaAgentRequest,
    _procedural_memory_response,
    router,
)


def _routes_by_path() -> dict[str, set[str]]:
    routes: dict[str, set[str]] = {}
    for route in router.routes:
        if hasattr(route, "path"):
            routes.setdefault(route.path, set()).update(
                getattr(route, "methods", None) or set()
            )
    return routes


def _query_param_names(path: str, method: str) -> set[str]:
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


# ---------------------------------------------------------------------------
# Route surface
# ---------------------------------------------------------------------------


def test_every_endpoint_the_harness_calls_exists():
    routes = _routes_by_path()

    assert "POST" in routes.get("/clients/create_or_get", set())
    assert "PATCH" in routes.get("/clients/{client_id}", set())
    assert "POST" in routes.get("/users/create_or_get", set())
    assert "GET" in routes.get("/agents", set())
    assert "POST" in routes.get("/agents/meta/initialize", set())
    assert "GET" in routes.get("/memory/search", set())
    assert "POST" in routes.get("/memory/add_sync", set())
    assert "POST" in routes.get("/memory/auto_dream", set())


def test_endpoints_accept_the_query_params_the_adapter_sends():
    # search_skills sends exactly these six params.
    assert {
        "memory_type",
        "query",
        "limit",
        "search_field",
        "search_method",
        "user_id",
    } <= _query_param_names("/memory/search", "GET")

    # auto_dream targets the eval user via ?user_id=...
    assert "user_id" in _query_param_names("/memory/auto_dream", "POST")

    # list_agents sends ?limit=1000.
    assert "limit" in _query_param_names("/agents", "GET")


def test_agents_response_rows_carry_id_and_agent_type():
    # list_agents selects the meta agent by agent_type and reads its id.
    assert "id" in AgentState.model_fields
    assert "agent_type" in AgentState.model_fields


# ---------------------------------------------------------------------------
# Ingest path
# ---------------------------------------------------------------------------


def test_episode_ingest_payload_validates_against_server_request_model():
    payload = build_add_sync_payload(
        meta_agent_id="agent-123",
        user_id="alfworld-run-20260712",
        session_id=build_episode_session_id("run-20260712", 3),
        user_content="Task: put a clean mug on the desk",
        assistant_content="I rinsed the mug and placed it on the desk.",
    )

    # Pydantic ignores unknown fields, so key membership must be explicit or a
    # server-side field removal would leave this test vacuously green.
    unknown = set(payload) - set(AddMemoryRequest.model_fields)
    assert not unknown, f"server request model dropped fields: {unknown}"

    req = AddMemoryRequest(**payload)

    assert req.user_id == "alfworld-run-20260712"
    assert req.session_id == "alfworld-run-20260712-ep-0003"
    assert req.chaining is True
    assert req.use_cache is True
    assert req.filter_tags is None  # server mirrors session_id itself
    assert [m["role"] for m in req.messages] == ["user", "assistant"]


def test_minted_session_ids_pass_the_server_validator():
    for session_id in (
        build_episode_session_id("run-20260712", 0),
        build_episode_session_id("Run With Spaces!", 9999),  # sanitized upstream
        build_consolidation_session_id("run-20260712", 5),
        build_consolidation_session_id("run-20260712", 10),
    ):
        AddMemoryRequest(
            meta_agent_id="agent-123",
            messages=[{"role": "user", "content": "x"}],
            session_id=session_id,
        )


# ---------------------------------------------------------------------------
# Consolidation path: /memory/auto_dream request + response models
# ---------------------------------------------------------------------------


def test_auto_dream_request_accepts_harness_body():
    # Exact body from MirixALFWorldAdapter.auto_dream (model omitted when None).
    req = AutoDreamRequest(mode="procedural", last_n_sessions=5, dry_run=False)
    assert req.mode == "procedural"
    assert req.last_n_sessions == 5

    # --consolidation-model adds a "model" field when set.
    AutoDreamRequest(
        mode="procedural", last_n_sessions=2, dry_run=False, model="openai/gpt-5.2"
    )

    # meta_agent_id stays optional: the harness relies on the server falling
    # back to the client's meta agent.
    assert "meta_agent_id" in AutoDreamRequest.model_fields
    assert AutoDreamRequest.model_fields["meta_agent_id"].default is None


def test_auto_dream_request_rejects_unknown_mode():
    with pytest.raises(ValidationError):
        AutoDreamRequest(mode="not-a-mode")


def test_auto_dream_response_carries_the_fields_the_runner_reads():
    fields = AutoDreamResponse.model_fields
    # runner.consolidate reads skills_changed + message off the response.
    assert "skills_changed" in fields
    assert "message" in fields
    # changes{created,edited,deleted} is stored verbatim in run artifacts.
    assert "changes" in fields


# ---------------------------------------------------------------------------
# Retrieval path: skill rows must carry what prompts.format_skill_knowledge uses
# ---------------------------------------------------------------------------


def test_procedural_search_row_has_prompt_fields():
    item = ProceduralMemoryItem(
        name="clean-then-place",
        entry_type="workflow",
        description="Clean an object before placing it",
        instructions="1. Take object\n2. Clean at sink\n3. Place at target",
        user_id="user-eval",
        organization_id="org-eval",
    )

    row = _procedural_memory_response(item)

    for key in ("name", "description", "instructions"):
        assert key in row, f"search row lost the '{key}' field"


# ---------------------------------------------------------------------------
# Bootstrap models
# ---------------------------------------------------------------------------


def test_client_bootstrap_models_accept_harness_payloads():
    create_body = {
        "client_id": "client-00000000-0000-4000-8000-000000000000",
        "org_id": "org-00000000-0000-4000-8000-000000000000",
        "name": "client-00000000-0000-4000-8000-000000000000",
        "write_scope": "admin",
        "read_scopes": ["admin"],
        "status": "active",
    }
    # Explicit key membership: Pydantic would silently ignore unknown fields.
    assert set(create_body) <= set(CreateOrGetClientRequest.model_fields)
    created = CreateOrGetClientRequest(**create_body)
    assert created.write_scope == "admin"

    patch_body = {
        "id": "client-00000000-0000-4000-8000-000000000000",
        "write_scope": "admin",
        "read_scopes": ["admin"],
        "status": "active",
    }
    assert set(patch_body) <= set(ClientUpdate.model_fields)
    patched = ClientUpdate(**patch_body)
    assert patched.write_scope == "admin"

    assert "write_scope" in Client.model_fields


def test_meta_agent_config_matches_server_config_schemas():
    config = build_meta_agent_config(model="openai/gpt-5.2", api_key="sk-test")

    InitializeMetaAgentRequest(config=config, update_agents=True)
    LLMConfig(**config["llm_config"])
    EmbeddingConfig(**config["embedding_config"])  # pins the openrouter literal
