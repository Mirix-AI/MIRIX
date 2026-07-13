"""Static wire-contract pins between the MetaClaw harness and the MIRIX server.

The harness talks to MIRIX only over HTTP, so nothing at import time forces the
two sides to stay compatible. These tests close that gap without a running
server: they import the REAL server router and Pydantic models and assert every
request/response feature the harness relies on. If someone changes the server
API, this suite fails at review time instead of an eval run failing hours in.

Pinned against:
- ``evals/metaclaw/mirix_adapters/generic_adapter.py`` (ingest path)
- ``evals/metaclaw/mirix_adapters/skills_adapter.py`` (retrieval path)
- ``evals/metaclaw/runner.py`` MIRIX prelude (client/user/meta-agent bootstrap)
"""

import asyncio
import os

import pytest
from pydantic import ValidationError

from evals.metaclaw.mirix_adapters.generic_adapter import (
    IN_BAND_TRIGGER_DISABLED_MIN,
    build_add_sync_payload,
)
from evals.metaclaw.mirix_adapters.skills_adapter import _mirix_skill_to_paper
from evals.testing import endpoint_source, query_param_names, routes_by_path
from mirix.schemas.agent import AgentState
from mirix.schemas.client import Client, ClientUpdate
from mirix.schemas.procedural_memory import ProceduralMemoryItem
from mirix.schemas.user import User
from mirix.server.rest_api import (
    AddMemoryRequest,
    CreateOrGetClientRequest,
    InitializeMetaAgentRequest,
    _procedural_memory_response,
    health_check,
)


# ---------------------------------------------------------------------------
# Route surface
# ---------------------------------------------------------------------------


def test_every_endpoint_the_harness_calls_exists():
    """One assert per HTTP call site in runner.py / the two adapters."""
    routes = routes_by_path()

    assert "GET" in routes.get("/health", set())  # runner + adapter preflight
    assert "POST" in routes.get("/users/create_or_get", set())
    assert "PATCH" in routes.get("/clients/{client_id}", set())
    assert "POST" in routes.get("/clients/create_or_get", set())
    assert "POST" in routes.get("/agents/meta/initialize", set())
    assert "GET" in routes.get("/agents", set())
    assert "POST" in routes.get("/memory/add_sync", set())  # openapi.json probe too
    assert "GET" in routes.get("/memory/search", set())


def test_search_and_agents_accept_the_query_params_the_adapters_send():
    search_params = query_param_names("/memory/search", "GET")
    # skills_adapter.retrieve / _fetch_all_skills_paper_shape send exactly these.
    assert {
        "memory_type",
        "query",
        "limit",
        "search_field",
        "search_method",
        "user_id",
    } <= search_params

    # generic_adapter._resolve_meta_agent_id sends ?limit=100.
    assert "limit" in query_param_names("/agents", "GET")


def test_search_envelope_still_exposes_the_results_key():
    """skills_adapter reads payload["results"]; the envelope is a handler-built
    dict with no response model, so pin the key at handler-source level.

    Pin the success-path assignment specifically — the bare key '"results"'
    also appears in the handler's error envelopes, which would keep a loose
    pin green even after the success envelope dropped the key."""
    assert '"results": all_results' in endpoint_source("search_memory")


def test_agents_response_rows_carry_id_and_agent_type():
    # Both the runner and generic_adapter select the meta agent by these keys.
    assert "id" in AgentState.model_fields
    assert "agent_type" in AgentState.model_fields


# ---------------------------------------------------------------------------
# Ingest path: build_add_sync_payload must validate as the server request model
# ---------------------------------------------------------------------------


def test_add_sync_payload_validates_against_server_request_model():
    payload = build_add_sync_payload(
        meta_agent_id="agent-123",
        user_id="eval-metaclaw-20260712-mirix-generic",
        query="How do I rotate the API keys?",
        answer="Use the rotation runbook.",
        session_id="day01-r3",
    )

    # Pydantic ignores unknown fields, so key membership must be explicit or a
    # server-side field removal would leave this test vacuously green.
    unknown = set(payload) - set(AddMemoryRequest.model_fields)
    assert not unknown, f"server request model dropped fields: {unknown}"

    req = AddMemoryRequest(**payload)

    assert req.meta_agent_id == "agent-123"
    assert req.user_id == "eval-metaclaw-20260712-mirix-generic"
    assert req.session_id == "day01-r3"
    assert req.chaining is True
    assert req.use_cache is True
    assert [m["role"] for m in req.messages] == ["user", "assistant"]


def test_add_sync_session_id_shape_is_accepted_by_server_validator():
    # The adapter mints f"{day}-r{round_index}" ids; pin the server's charset.
    for session_id in ("day01-r3", "day30-r12", "a" * 64):
        AddMemoryRequest(
            meta_agent_id="agent-123",
            messages=[{"role": "user", "content": "x"}],
            session_id=session_id,
        )
    for bad in ("", "day 01-r3", "a" * 65):
        with pytest.raises(ValidationError):
            AddMemoryRequest(
                meta_agent_id="agent-123",
                messages=[{"role": "user", "content": "x"}],
                session_id=bad,
            )


def test_server_rejects_conflicting_session_id_and_filter_tags():
    """The adapter deliberately omits filter_tags because the server mirrors
    session_id into it; pin the agreement validator that motivates that."""
    with pytest.raises(ValidationError):
        AddMemoryRequest(
            meta_agent_id="agent-123",
            messages=[{"role": "user", "content": "x"}],
            session_id="day01-r3",
            filter_tags={"session_id": "day01-r4"},
        )


# ---------------------------------------------------------------------------
# Preflight: /health must expose the in-band trigger threshold
# ---------------------------------------------------------------------------


def test_health_exposes_skill_trigger_threshold():
    body = asyncio.run(health_check())

    threshold = body["skill_trigger_session_threshold"]
    assert isinstance(threshold, int)
    assert threshold > 0
    if "SKILL_TRIGGER_SESSION_THRESHOLD" not in os.environ:
        # By default the in-band trigger must be enabled; the adapter refuses
        # to run otherwise. Skipped when the env deliberately overrides it.
        assert threshold < IN_BAND_TRIGGER_DISABLED_MIN


# ---------------------------------------------------------------------------
# Retrieval path: procedural search rows must map to the paper skill shape
# ---------------------------------------------------------------------------


def test_procedural_search_row_maps_to_paper_skill_shape():
    item = ProceduralMemoryItem(
        name="rotate-api-keys",
        entry_type="workflow",
        description="Rotate the API keys safely",
        instructions="1. Freeze traffic\n2. Rotate\n3. Verify",
        user_id="user-eval",
        organization_id="org-eval",
    )

    row = _procedural_memory_response(item)
    for key in (
        "memory_type",
        "id",
        "entry_type",
        "name",
        "description",
        "instructions",
    ):
        assert key in row, f"search row lost the '{key}' field"

    paper = _mirix_skill_to_paper(row)
    assert paper["name"] == "rotate-api-keys"
    assert paper["description"] == "Rotate the API keys safely"
    assert paper["content"] == item.instructions  # the load-bearing rename
    assert paper["category"] == "workflow"


# ---------------------------------------------------------------------------
# Bootstrap: client / user / meta-agent request+response models
# ---------------------------------------------------------------------------


def test_client_bootstrap_models_accept_runner_payloads():
    # POST /clients/create_or_get body (runner.py fallback path)
    create_body = {
        "client_id": "client-00000000-0000-4000-8000-000000000000",
        "org_id": "org-00000000-0000-4000-8000-000000000000",
        "write_scope": "admin",
    }
    assert set(create_body) <= set(CreateOrGetClientRequest.model_fields)
    created = CreateOrGetClientRequest(**create_body)
    assert created.write_scope == "admin"

    # PATCH /clients/{client_id} body (runner.py primary path)
    patch_body = {
        "id": "client-00000000-0000-4000-8000-000000000000",
        "write_scope": "admin",
    }
    assert set(patch_body) <= set(ClientUpdate.model_fields)
    patched = ClientUpdate(**patch_body)
    assert patched.write_scope == "admin"

    # The runner reads write_scope off the response to confirm admin scope.
    assert "write_scope" in Client.model_fields


def test_user_response_carries_id():
    assert "id" in User.model_fields


def test_meta_initialize_request_accepts_runner_payload():
    InitializeMetaAgentRequest(
        config={
            "llm_config": {
                "model": "openai/gpt-5.2",
                "model_endpoint_type": "openai",
                "model_endpoint": "https://openrouter.ai/api/v1",
                "context_window": 128000,
            },
            "embedding_config": {
                "embedding_model": "gemini-embedding-001",
                "embedding_endpoint_type": "openrouter",
                "embedding_endpoint": "https://openrouter.ai/api/v1",
                "embedding_dim": 4096,
                "embedding_chunk_size": 2048,
            },
        },
        update_agents=True,
    )
