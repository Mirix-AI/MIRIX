"""Regression coverage for MIRIX generic tool-transcript capture."""

from __future__ import annotations

import queue
import tempfile
import threading
import types

import httpx
import pytest

from evals.metaclaw.vendor.metaclaw import api_server
from evals.metaclaw.vendor.metaclaw.config import MetaClawConfig


class _CaptureEvolver:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def distill_round(self, **kwargs):
        self.calls.append(kwargs)
        return {"ok": True, "evolved": False}


@pytest.mark.asyncio
async def test_generic_memory_ingest_consumes_proxy_captured_tool_transcript(monkeypatch):
    """A real proxy chat request with tool_calls/tool result reaches ingest_round.

    This is the important MetaClaw boundary for OpenClaw tool loops: after a tool
    call, the next model request contains the previous assistant ``tool_calls``
    plus the ``role=tool`` result. The generic MIRIX ingest endpoint must pass
    that captured transcript to the memory adapter.
    """
    monkeypatch.setattr(
        api_server.MetaClawAPIServer,
        "_load_tokenizer",
        lambda self: None,
    )

    evolver = _CaptureEvolver()
    cfg = MetaClawConfig(
        mode="skills_only",
        skill_evolution_mode="generic_memory",
        enable_skill_evolution=True,
        record_enabled=False,
        record_dir=tempfile.mkdtemp(prefix="metaclaw-transcript-test-"),
        llm_model_id="fake-model",
        llm_api_base="http://fake.invalid/v1",
        llm_api_key="sk-test",
    )
    server = api_server.MetaClawAPIServer(
        config=cfg,
        output_queue=queue.Queue(),
        submission_enabled=threading.Event(),
        skill_evolver=evolver,
    )
    server.submission_enabled.set()

    async def fake_forward_to_llm(self, body, session_id):
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Final answer after reading the file.",
                    }
                }
            ]
        }

    server._forward_to_llm = types.MethodType(fake_forward_to_llm, server)

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        chat_resp = await client.post(
            "/v1/chat/completions",
            headers={"X-Session-Id": "session-tool-1", "X-Turn-Type": "main"},
            json={
                "model": "fake-model",
                "messages": [
                    {"role": "user", "content": "Read notes.md and answer."},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_read_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":"notes.md"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_read_1",
                        "name": "read_file",
                        "content": "notes.md says: use +08:00 timestamps.",
                    },
                ],
            },
        )
        ingest_resp = await client.post(
            "/v1/memory/ingest_round",
            json={
                "session_id": "session-tool-1",
                "day": "day01",
                "round_id": "r1",
                "round_index": 1,
                "query": "Read notes.md and answer.",
                "answer": "Final answer after reading the file.",
                "session_done": False,
            },
        )

    assert chat_resp.status_code == 200, chat_resp.text
    assert ingest_resp.status_code == 200, ingest_resp.text
    assert evolver.calls

    transcript = evolver.calls[0]["transcript"]
    assert transcript["source"] == "metaclaw_proxy"
    messages = transcript["messages"]
    assert [m.get("role") for m in messages] == ["user", "assistant", "tool"]
    assert messages[1]["tool_calls"][0]["function"]["name"] == "read_file"
    assert messages[2]["content"] == "notes.md says: use +08:00 timestamps."


@pytest.mark.asyncio
async def test_generic_memory_ingest_uses_recent_tui_transcript_when_session_header_missing(monkeypatch):
    """Real OpenClaw CLI traffic may omit X-Session-Id on chat requests."""
    monkeypatch.setattr(
        api_server.MetaClawAPIServer,
        "_load_tokenizer",
        lambda self: None,
    )

    evolver = _CaptureEvolver()
    cfg = MetaClawConfig(
        mode="skills_only",
        skill_evolution_mode="generic_memory",
        enable_skill_evolution=True,
        record_enabled=False,
        record_dir=tempfile.mkdtemp(prefix="metaclaw-transcript-test-"),
        llm_model_id="fake-model",
        llm_api_base="http://fake.invalid/v1",
        llm_api_key="sk-test",
    )
    server = api_server.MetaClawAPIServer(
        config=cfg,
        output_queue=queue.Queue(),
        submission_enabled=threading.Event(),
        skill_evolver=evolver,
    )
    server.submission_enabled.set()

    async def fake_forward_to_llm(self, body, session_id):
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Final answer after reading the file.",
                    }
                }
            ]
        }

    server._forward_to_llm = types.MethodType(fake_forward_to_llm, server)

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        chat_resp = await client.post(
            "/v1/chat/completions",
            headers={"X-Turn-Type": "main"},
            json={
                "model": "fake-model",
                "turn_type": "main",
                "messages": [
                    {"role": "user", "content": "Read notes.md and answer."},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_read_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":"notes.md"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_read_1",
                        "name": "read_file",
                        "content": "notes.md says: use +08:00 timestamps.",
                    },
                ],
            },
        )
        ingest_resp = await client.post(
            "/v1/memory/ingest_round",
            json={
                "session_id": "day01-real-bench-session",
                "day": "day01",
                "round_id": "r1",
                "round_index": 1,
                "query": "Read notes.md and answer.",
                "answer": "Final answer after reading the file.",
                "session_done": False,
            },
        )

    assert chat_resp.status_code == 200, chat_resp.text
    assert ingest_resp.status_code == 200, ingest_resp.text
    assert evolver.calls

    transcript = evolver.calls[0]["transcript"]
    assert transcript["source"] == "metaclaw_proxy"
    messages = transcript["messages"]
    assert [m.get("role") for m in messages] == ["user", "assistant", "tool"]
    assert messages[1]["tool_calls"][0]["function"]["name"] == "read_file"
    assert messages[2]["content"] == "notes.md says: use +08:00 timestamps."
