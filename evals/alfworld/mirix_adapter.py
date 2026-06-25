"""MIRIX REST adapter for ALFWorld online procedural-memory evals."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CLIENT_ID = "client-00000000-0000-4000-8000-000000000000"
DEFAULT_ORG_ID = "org-00000000-0000-4000-8000-000000000000"
DEFAULT_TIMEOUT_S = 1800.0
DEFAULT_MIRIX_URL = "http://127.0.0.1:8531"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
META_AGENT_TYPE = "meta_memory_agent"
SESSION_ID_MAX_LEN = 64
MEMORY_MAX_DETAILED_STEPS = 12
MEMORY_OBSERVATION_CHARS = 300
MEMORY_MODEL_RESPONSE_CHARS = 300
_SESSION_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_-]+")


class MirixAdapterError(RuntimeError):
    """Raised for MIRIX setup or REST contract failures."""


def build_meta_agent_config(
    *,
    api_key: str,
    model: str = "openai/gpt-5.2",
    model_endpoint: str = OPENROUTER_BASE_URL,
    embedding_model: str = "gemini-embedding-001",
    embedding_endpoint: str = OPENROUTER_BASE_URL,
    embedding_dim: int = 4096,
    embedding_chunk_size: int = 2048,
) -> dict[str, Any]:
    """Build the config body accepted by ``POST /agents/meta/initialize``."""

    return {
        "llm_config": {
            "model": model,
            "model_endpoint_type": "openai",
            "model_endpoint": model_endpoint,
            "context_window": 128000,
            "api_key": api_key,
        },
        "embedding_config": {
            "embedding_model": embedding_model,
            "embedding_endpoint_type": "openrouter",
            "embedding_endpoint": embedding_endpoint,
            "embedding_dim": embedding_dim,
            "embedding_chunk_size": embedding_chunk_size,
            "api_key": api_key,
        },
    }


def build_add_sync_payload(
    *,
    meta_agent_id: str,
    user_id: str,
    session_id: str,
    user_content: str,
    assistant_content: str,
) -> dict[str, Any]:
    """Build the exact body used for one ALFWorld session ingest."""

    return {
        "meta_agent_id": meta_agent_id,
        "user_id": user_id,
        "session_id": session_id,
        "chaining": True,
        "use_cache": True,
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    }


def build_episode_session_id(run_id: str, episode_index: int) -> str:
    """Return a MIRIX-compatible session id for one ALFWorld episode."""

    return _build_session_id(run_id, f"ep-{episode_index:04d}")


def build_consolidation_session_id(run_id: str) -> str:
    """Return a MIRIX-compatible sentinel session id for consolidation."""

    return _build_session_id(run_id, "boundary")


class MirixALFWorldAdapter:
    """Small synchronous adapter over MIRIX's production memory endpoints."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_MIRIX_URL,
        user_id: str,
        client_id: str = DEFAULT_CLIENT_ID,
        org_id: str = DEFAULT_ORG_ID,
        meta_agent_id: str | None = None,
        authorization: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        transport: Any = None,
    ) -> None:
        try:
            import httpx
        except Exception as exc:  # pragma: no cover - optional in bare env
            raise MirixAdapterError(
                "httpx is required for MIRIX REST access. Install eval dependencies "
                'with `pip install -e ".[eval]"`.'
            ) from exc

        self.base_url = base_url.rstrip("/")
        self.user_name = user_id
        self.client_id = client_id
        self.org_id = org_id
        self.meta_agent_id = meta_agent_id
        self.authorization = authorization
        self._resolved_user_id: str | None = None

        headers = {
            "X-Client-Id": self.client_id,
            "X-Org-Id": self.org_id,
            "Content-Type": "application/json",
        }
        if authorization:
            headers["Authorization"] = authorization

        kwargs: dict[str, Any] = {
            "base_url": self.base_url,
            "timeout": timeout_s,
            "headers": headers,
        }
        if transport is not None:
            kwargs["transport"] = transport
        self._http = httpx.Client(**kwargs)

    @property
    def resolved_user_id(self) -> str:
        return self._resolved_user_id or self.user_name

    def close(self) -> None:
        self._http.close()

    def prepare(
        self,
        *,
        init_meta_agent: bool = False,
        init_api_key: str | None = None,
        init_model: str = "openai/gpt-5.2",
    ) -> dict[str, Any]:
        """Ensure client/user/meta-agent state needed by the eval exists."""

        client = self.ensure_writable_client()
        user_id = self.ensure_user_id()
        meta_agent_id = self.ensure_meta_agent_id(
            auto_initialize=init_meta_agent,
            init_api_key=init_api_key,
            init_model=init_model,
        )
        return {"client": client, "user_id": user_id, "meta_agent_id": meta_agent_id}

    def ensure_writable_client(self) -> dict[str, Any]:
        """Create the eval client if needed and ensure it can write/read admin scope."""

        body = {
            "client_id": self.client_id,
            "org_id": self.org_id,
            "name": self.client_id,
            "write_scope": "admin",
            "read_scopes": ["admin"],
            "status": "active",
        }
        resp = self._http.post("/clients/create_or_get", json=body)
        _raise_for_status(resp, "create_or_get client")

        patch_body = {
            "id": self.client_id,
            "write_scope": "admin",
            "read_scopes": ["admin"],
            "status": "active",
        }
        resp = self._http.patch(f"/clients/{self.client_id}", json=patch_body)
        _raise_for_status(resp, "update client write/read scope")
        return _json(resp)

    def ensure_user_id(self) -> str:
        if self._resolved_user_id:
            return self._resolved_user_id
        resp = self._http.post(
            "/users/create_or_get",
            json={"user_id": self.user_name, "name": self.user_name},
        )
        _raise_for_status(resp, "create_or_get user")
        payload = _json(resp)
        self._resolved_user_id = str(payload.get("id") or self.user_name)
        return self._resolved_user_id

    def list_agents(self) -> list[dict[str, Any]]:
        resp = self._http.get("/agents", params={"limit": 1000})
        _raise_for_status(resp, "list agents")
        payload = _json(resp)
        if not isinstance(payload, list):
            raise MirixAdapterError(f"Expected /agents list response, got: {payload!r}")
        return [agent for agent in payload if isinstance(agent, dict)]

    def ensure_meta_agent_id(
        self,
        *,
        auto_initialize: bool = False,
        init_api_key: str | None = None,
        init_model: str = "openai/gpt-5.2",
    ) -> str:
        if self.meta_agent_id:
            return self.meta_agent_id

        for agent in self.list_agents():
            if agent.get("agent_type") == META_AGENT_TYPE:
                self.meta_agent_id = str(agent["id"])
                return self.meta_agent_id

        if auto_initialize:
            api_key = init_api_key or read_first_env_key(
                "OPENROUTER_API_KEY", "OPENAI_API_KEY"
            )
            if not api_key:
                raise MirixAdapterError(
                    "No meta agent found and no OPENROUTER_API_KEY/OPENAI_API_KEY "
                    "is available for --init-meta-agent."
                )
            payload = self.initialize_meta_agent(api_key=api_key, model=init_model)
            if payload and payload.get("id"):
                self.meta_agent_id = str(payload["id"])
                return self.meta_agent_id

        raise MirixAdapterError(
            "No MIRIX meta_memory_agent found for this client. Run "
            "`python -m evals.metaclaw.init_meta_agent` first, or pass "
            "`--init-meta-agent` to the ALFWorld runner."
        )

    def initialize_meta_agent(
        self,
        *,
        api_key: str,
        model: str = "openai/gpt-5.2",
        update_agents: bool = True,
    ) -> dict[str, Any]:
        body = {
            "config": build_meta_agent_config(api_key=api_key, model=model),
            "update_agents": update_agents,
        }
        resp = self._http.post("/agents/meta/initialize", json=body)
        _raise_for_status(resp, "initialize meta agent")
        payload = _json(resp)
        if payload is None:
            raise MirixAdapterError(
                "/agents/meta/initialize returned null; the client is likely read-only."
            )
        return payload

    def search_skills(self, query: str, *, top_k: int = 6) -> list[dict[str, Any]]:
        """Retrieve procedural memories for the current ALFWorld task."""

        user_id = self.ensure_user_id()
        resp = self._http.get(
            "/memory/search",
            params={
                "memory_type": "procedural",
                "query": query,
                "limit": top_k,
                "search_field": "description",
                "search_method": "",
                "user_id": user_id,
            },
        )
        _raise_for_status(resp, "search procedural memory")
        payload = _json(resp)
        rows = payload.get("results") if isinstance(payload, dict) else payload
        return [row for row in (rows or []) if isinstance(row, dict)]

    def ingest_session(
        self,
        *,
        session_id: str,
        user_content: str,
        assistant_content: str,
    ) -> dict[str, Any]:
        meta_agent_id = self.ensure_meta_agent_id()
        user_id = self.ensure_user_id()
        payload = build_add_sync_payload(
            meta_agent_id=meta_agent_id,
            user_id=user_id,
            session_id=session_id,
            user_content=user_content,
            assistant_content=assistant_content,
        )
        resp = self._http.post("/memory/add_sync", json=payload)
        _raise_for_status(resp, f"add_sync session {session_id}")
        return _json(resp)

    def seal_for_consolidation(self, *, run_id: str) -> dict[str, Any]:
        """Write a constant sentinel session so the latest real episode is sealed."""

        return self.ingest_session(
            session_id=build_consolidation_session_id(run_id),
            user_content=(
                "ALFWorld consolidation boundary. This is a sentinel turn used "
                "only to close the latest real episode session."
            ),
            assistant_content="Boundary marker only; ignore for task strategy.",
        )

    def auto_dream(
        self,
        *,
        last_n_sessions: int = 5,
        model: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "mode": "procedural",
            "last_n_sessions": last_n_sessions,
            "dry_run": dry_run,
        }
        if model:
            body["model"] = model
        resp = self._http.post(
            "/memory/auto_dream",
            params={"user_id": self.ensure_user_id()},
            json=body,
        )
        _raise_for_status(resp, "auto_dream procedural")
        return _json(resp)


def read_first_env_key(*names: str) -> str | None:
    """Read the first configured API key from env or the repo ``.env`` file."""

    for name in names:
        value = os.environ.get(name)
        if value:
            return value

    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return None
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() in names:
            return value.strip().strip('"').strip("'")
    return None


def _build_session_id(run_id: str, suffix: str) -> str:
    safe_suffix = _sanitize_session_component(suffix, max_len=16)
    max_run_len = SESSION_ID_MAX_LEN - len("alfworld--") - len(safe_suffix)
    safe_run_id = _sanitize_session_component(run_id, max_len=max_run_len)
    return f"alfworld-{safe_run_id}-{safe_suffix}"


def _sanitize_session_component(value: str, *, max_len: int) -> str:
    safe = _SESSION_COMPONENT_RE.sub("_", str(value)).strip("_-")
    safe = safe[:max_len].strip("_-")
    return safe or "run"


def render_episode_for_memory(episode: Mapping[str, Any]) -> tuple[str, str]:
    """Convert one episode record into two agent-visible memory messages."""

    steps = episode.get("steps") or []
    lines = [
        f"Episode id: {episode.get('id', '')}",
        f"Task type: {episode.get('task_type', '')}",
        f"Task: {episode.get('task_description', '')}",
        f"Gamefile: {episode.get('gamefile', '')}",
        "",
        _render_action_sequence(steps),
        "",
        "Trajectory:",
    ]
    detailed_steps = _select_memory_steps(steps)
    for step in detailed_steps:
        if step is None:
            lines.append(f"[... skipped middle steps; total steps: {len(steps)} ...]")
            lines.append("")
        else:
            lines.extend(_render_step_for_memory(step))

    assistant = (
        f"Episode result: {'success' if episode.get('success') else 'failure'}.\n"
        f"Steps used: {episode.get('n_steps', len(steps))}.\n"
        f"Failure reason: {episode.get('fail_reason') or ''}"
    )
    return _clip("\n".join(lines), 20000), assistant


def _render_action_sequence(steps: Any) -> str:
    if not isinstance(steps, list) or not steps:
        return "Action sequence: <empty>"
    actions = []
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        suffix = ""
        if step.get("won"):
            suffix = "[won]"
        elif step.get("done"):
            suffix = "[done]"
        actions.append(f"{step.get('step')}:{step.get('action', '')}{suffix}")
    return _clip("Action sequence: " + " -> ".join(actions), 5000)


def _select_memory_steps(steps: Any) -> list[Mapping[str, Any] | None]:
    if not isinstance(steps, list):
        return []
    clean_steps = [step for step in steps if isinstance(step, Mapping)]
    if len(clean_steps) <= MEMORY_MAX_DETAILED_STEPS:
        return clean_steps
    head = MEMORY_MAX_DETAILED_STEPS // 2
    tail = MEMORY_MAX_DETAILED_STEPS - head
    return [*clean_steps[:head], None, *clean_steps[-tail:]]


def _render_step_for_memory(step: Mapping[str, Any]) -> list[str]:
    return [
        f"Step {step.get('step')}:",
        f"Observation: {_clip(str(step.get('observation', '')), MEMORY_OBSERVATION_CHARS)}",
        "Admissible actions: "
        + _clip(
            json.dumps(step.get("admissible_actions", []), ensure_ascii=False),
            MEMORY_OBSERVATION_CHARS,
        ),
        f"Model response: {_clip(str(step.get('model_response', '')), MEMORY_MODEL_RESPONSE_CHARS)}",
        f"Executed action: {step.get('action', '')}",
        f"Environment feedback: {_clip(str(step.get('next_observation', '')), MEMORY_OBSERVATION_CHARS)}",
        f"Reward: {step.get('reward', 0.0)} Done: {step.get('done', False)}",
        "",
    ]


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    marker = "\n[... clipped ...]\n"
    head = max_chars // 2
    tail = max_chars - head - len(marker)
    return text[:head] + marker + text[-tail:]


def _raise_for_status(resp: Any, label: str) -> None:
    try:
        resp.raise_for_status()
    except Exception as exc:
        text = getattr(resp, "text", "")
        raise MirixAdapterError(f"MIRIX {label} failed: {text[:1000]}") from exc


def _json(resp: Any) -> Any:
    try:
        return resp.json()
    except ValueError as exc:
        raise MirixAdapterError(f"Expected JSON response, got: {resp.text[:1000]}") from exc
