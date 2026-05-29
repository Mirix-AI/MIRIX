"""Pytest fixtures for offline integration smoke tests.

These fixtures spin up tiny FastAPI servers (LLM-shaped and MIRIX-shaped) in
background threads so the metaclaw runner can be exercised end-to-end without
spawning real ``clawdbot`` / ``openclaw`` processes, without hitting OpenRouter,
and without requiring a real MIRIX server.

CRITICAL: smoke tests MUST NOT spawn the real metaclaw proxy or the real bench
subprocess.  The runner exposes ``proxy_starter`` / ``proxy_stopper`` /
``bench_runner`` DI hooks; the smoke tests inject stubs that write a
deterministic ``report.json`` so the runner's parsing path is exercised but no
clawdbot is touched.
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from typing import Iterator

import pytest
import uvicorn
from fastapi import FastAPI, Request


def _preallocate_listener() -> socket.socket:
    """Bind a TCP socket to a kernel-chosen free port on 127.0.0.1.

    Returned socket is already listening; pass it to uvicorn via
    ``sockets=[sock]`` to avoid the classic free-port race (where another
    process can grab the port between ``close()`` and uvicorn's ``bind()``).
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    s.listen(128)
    s.setblocking(False)
    return s


class _BackgroundUvicorn:
    """Run a uvicorn Server in a background thread; ``stop()`` shuts it down.

    We pre-bind the listening socket on the main thread (`_preallocate_listener`)
    and hand it to uvicorn's ``server.run(sockets=[sock])`` so there is no
    free-port race: by the time we yield to the test, the port is unambiguously
    owned by this uvicorn instance.
    """

    def __init__(self, app: FastAPI):
        self._sock = _preallocate_listener()
        self.port = self._sock.getsockname()[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        config = uvicorn.Config(
            app=app,
            log_level="warning",
            lifespan="off",
            access_log=False,
        )
        self.server = uvicorn.Server(config)
        # Defeat uvicorn's signal handler install — we're not the main thread.
        self.server.install_signal_handlers = lambda: None  # type: ignore[assignment]
        self._thread_exc: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            self.server.run(sockets=[self._sock])
        except BaseException as e:  # noqa: BLE001 — surface in start()
            self._thread_exc = e

    def start(self, ready_timeout_s: float = 10.0) -> None:
        self.thread.start()
        deadline = time.time() + ready_timeout_s
        while time.time() < deadline:
            if self._thread_exc is not None:
                # uvicorn died on startup — propagate the original exception.
                raise RuntimeError(
                    f"stub server on port {self.port} crashed during startup: "
                    f"{type(self._thread_exc).__name__}: {self._thread_exc}"
                ) from self._thread_exc
            if self.server.started:
                return
            time.sleep(0.05)
        raise RuntimeError(
            f"stub server on port {self.port} did not become ready in {ready_timeout_s}s"
        )

    def stop(self, timeout_s: float = 5.0) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=timeout_s)
        if self.thread.is_alive():
            # Last-ditch: force-exit so a wedged uvicorn doesn't keep the port.
            self.server.force_exit = True
            self.thread.join(timeout=timeout_s)
        # Best-effort: close the listening socket if uvicorn somehow didn't.
        with contextlib.suppress(OSError):
            self._sock.close()


def _build_stub_llm_app() -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(req: Request):  # noqa: ARG001 — body inspected for echo
        body = await req.json()
        return {
            "id": "stub-001",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "stub-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "stub answer",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    return app


def _build_stub_mirix_app() -> FastAPI:
    """Tiny MIRIX-shaped server exposing only the endpoints adapters touch.

    Must satisfy:
      - runner._mirix_health_diagnose:        GET /health
      - runner._mirix_create_or_get_user:     POST /users/create_or_get
      - runner._mirix_reset_user_skills:      POST /v1/skills/reset
      - MirixSkillManagerAdapter:             GET /v1/skills, POST /v1/skills
      - MirixEvolverAdapter._preflight:       GET /openapi.json
      - MirixEvolverAdapter.evolve:           POST /v1/skills/evolve
    """
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/users/create_or_get")
    async def create_or_get(req: Request):
        body = await req.json()
        uid = body.get("user_id") or "stub-user"
        return {"id": uid, "name": body.get("name", uid)}

    @app.get("/v1/skills")
    async def list_skills():
        return {"skills": []}

    @app.post("/v1/skills")
    async def create_skill(req: Request):
        body = await req.json()
        return {"id": "skill-stub-001", **body}

    @app.post("/v1/skills/evolve")
    async def evolve(req: Request):  # noqa: ARG001
        return {
            "success": True,
            "changes": {"created": [], "edited": [], "deleted": []},
            "summary": {"created_count": 0, "edited_count": 0, "deleted_count": 0},
        }

    @app.post("/v1/skills/reset")
    async def reset():
        return {"success": True}

    # /openapi.json is provided automatically by FastAPI, but the evolver
    # adapter's preflight specifically looks for the /v1/skills/evolve path in
    # the openapi document — FastAPI's auto-generated schema includes it.

    return app


@pytest.fixture
def stub_llm() -> Iterator[str]:
    """Yield a base URL like ``http://127.0.0.1:<port>`` for the stub LLM."""
    srv = _BackgroundUvicorn(_build_stub_llm_app())
    try:
        srv.start()
        yield srv.base_url
    finally:
        with contextlib.suppress(Exception):
            srv.stop()


@pytest.fixture
def stub_mirix() -> Iterator[str]:
    """Yield a base URL like ``http://127.0.0.1:<port>`` for the stub MIRIX."""
    srv = _BackgroundUvicorn(_build_stub_mirix_app())
    try:
        srv.start()
        yield srv.base_url
    finally:
        with contextlib.suppress(Exception):
            srv.stop()
