"""Tests for MIRIX generic production-memory adapter."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable, List

import httpx
import pytest

from evals.metaclaw.mirix_adapters.generic_adapter import (
    ADD_SYNC_ENDPOINT_PATH,
    AGENTS_ENDPOINT_PATH,
    DISTILL_FORBIDDEN_KEYS,
    EVOLVE_FAILURE_MARKER,
    HEALTH_ENDPOINT_PATH,
    IN_BAND_TRIGGER_DISABLED_MIN,
    MirixGenericMemoryAdapter,
    build_add_sync_payload,
)


_META_AGENT_ID = "agent-meta-0001"

_OPENAPI_OK = {
    "openapi": "3.1.0",
    "paths": {
        ADD_SYNC_ENDPOINT_PATH: {"post": {}},
    },
}

_OPENAPI_MISSING = {"openapi": "3.1.0", "paths": {}}

_AGENTS_ROWS = [
    {"id": "agent-other", "agent_type": "episodic_memory_agent"},
    {"id": _META_AGENT_ID, "agent_type": "meta_memory_agent"},
]


def _make_handler(
    captured: List[httpx.Request],
    *,
    openapi=_OPENAPI_OK,
    agents=_AGENTS_ROWS,
    add_sync_body: dict | None = None,
    trigger_threshold: int = 5,
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        path = request.url.path
        if path == "/openapi.json":
            return httpx.Response(200, json=openapi)
        if path == HEALTH_ENDPOINT_PATH:
            return httpx.Response(
                200,
                json={
                    "status": "healthy",
                    "skill_trigger_session_threshold": trigger_threshold,
                    "message_retain_last_n_sessions": 5,
                },
            )
        if path == AGENTS_ENDPOINT_PATH:
            return httpx.Response(200, json=agents)
        if path == ADD_SYNC_ENDPOINT_PATH:
            return httpx.Response(
                200,
                json=add_sync_body
                or {"success": True, "status": "processed", "message_count": 2},
            )
        return httpx.Response(404, json={"error": f"unexpected {path}"})

    return handler


def _make_adapter(handler, **kw) -> MirixGenericMemoryAdapter:
    return MirixGenericMemoryAdapter(
        base_url="http://mock.test",
        user_id="user-1",
        transport=httpx.MockTransport(handler),
        **kw,
    )


def _json_body(request: httpx.Request) -> dict:
    return json.loads(request.content.decode("utf-8"))


def _keys(obj) -> set[str]:
    if isinstance(obj, dict):
        out = set(obj)
        for value in obj.values():
            out.update(_keys(value))
        return out
    if isinstance(obj, list):
        out: set[str] = set()
        for item in obj:
            out.update(_keys(item))
        return out
    return set()


def test_preflight_requires_add_sync_endpoint():
    captured: List[httpx.Request] = []
    handler = _make_handler(captured, openapi=_OPENAPI_MISSING)
    with pytest.raises(RuntimeError, match="add_sync"):
        _make_adapter(handler)


def test_preflight_raises_when_in_band_trigger_disabled():
    captured: List[httpx.Request] = []
    handler = _make_handler(
        captured,
        trigger_threshold=IN_BAND_TRIGGER_DISABLED_MIN,
    )
    with pytest.raises(RuntimeError, match="DISABLED"):
        _make_adapter(handler)


def test_build_add_sync_payload_has_only_visible_turns():
    body = build_add_sync_payload(
        meta_agent_id="agent-1",
        user_id="user-1",
        query="question with previous feedback",
        answer="final answer",
        session_id="day01-r1",
    )
    keys = _keys(body)
    for forbidden in DISTILL_FORBIDDEN_KEYS:
        assert forbidden not in keys
    assert body["meta_agent_id"] == "agent-1"
    assert body["user_id"] == "user-1"
    assert body["session_id"] == "day01-r1"
    assert body["messages"] == [
        {"role": "user", "content": "question with previous feedback"},
        {"role": "assistant", "content": "final answer"},
    ]


def test_build_add_sync_payload_can_include_sanitized_tool_transcript():
    transcript = {
        "inline_score": {"passed": False},
        "eval": {"answer": ["B"]},
        "messages": [
            {"role": "system", "content": "hidden injected skill block"},
            {"role": "user", "content": "question with previous feedback"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": "{\"path\":\"notes.md\"}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "read_file",
                "content": "file says created_at must be +08:00",
            },
        ],
    }

    body = build_add_sync_payload(
        meta_agent_id="agent-1",
        user_id="user-1",
        query="question with previous feedback",
        answer="final answer",
        session_id="day01-r1",
        transcript=transcript,
    )

    keys = _keys(body)
    for forbidden in DISTILL_FORBIDDEN_KEYS:
        assert forbidden not in keys
    user_content = body["messages"][0]["content"]
    assert "[MetaClaw agent-visible transcript]" in user_content
    assert "read_file" in user_content
    assert "file says created_at must be +08:00" in user_content
    assert "hidden injected skill block" not in user_content
    assert "inline_score" not in user_content
    assert "eval" not in user_content
    assert body["messages"][1] == {"role": "assistant", "content": "final answer"}


def test_distill_round_posts_only_add_sync_and_uses_per_turn_session_id():
    captured: List[httpx.Request] = []
    adapter = _make_adapter(_make_handler(captured))
    try:
        out = asyncio.run(
            adapter.distill_round(
                day="day01",
                round_id="round-1",
                round_index=3,
                query="q",
                answer="a",
                session_id="whole-day-session",
            )
        )
    finally:
        adapter.close()

    assert out["ok"] is True
    assert out["evolved"] is False
    assert out["session_id"] == "day01-r3"
    paths = [r.url.path for r in captured]
    assert ADD_SYNC_ENDPOINT_PATH in paths
    assert "/memory/auto_dream" not in paths
    assert not any(path.startswith("/v1/skills") for path in paths)
    add_sync = [r for r in captured if r.url.path == ADD_SYNC_ENDPOINT_PATH][0]
    body = _json_body(add_sync)
    assert body["meta_agent_id"] == _META_AGENT_ID
    assert body["session_id"] == "day01-r3"


def test_distill_round_forwards_tool_transcript_to_add_sync():
    captured: List[httpx.Request] = []
    adapter = _make_adapter(_make_handler(captured))
    try:
        out = asyncio.run(
            adapter.distill_round(
                day="day01",
                round_id="round-1",
                round_index=3,
                query="q",
                answer="a",
                transcript={
                    "messages": [
                        {"role": "user", "content": "q"},
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {
                                        "name": "list_dir",
                                        "arguments": "{\"path\":\".\"}",
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "call_1",
                            "name": "list_dir",
                            "content": "README.md",
                        },
                    ]
                },
            )
        )
    finally:
        adapter.close()

    assert out["ok"] is True
    add_sync = [r for r in captured if r.url.path == ADD_SYNC_ENDPOINT_PATH][0]
    user_content = _json_body(add_sync)["messages"][0]["content"]
    assert "list_dir" in user_content
    assert "README.md" in user_content


def test_session_done_does_not_call_mirix():
    captured: List[httpx.Request] = []
    adapter = _make_adapter(_make_handler(captured))
    try:
        before = len(captured)
        out = asyncio.run(
            adapter.distill_round(
                day="day01",
                round_id="done",
                round_index=0,
                query="",
                answer="",
                session_done=True,
            )
        )
    finally:
        adapter.close()
    assert out == {
        "ok": True,
        "evolved": False,
        "flush": True,
        "turns_ingested": 0,
    }
    assert len(captured) == before


def test_add_sync_semantic_failure_is_loud(caplog):
    captured: List[httpx.Request] = []
    adapter = _make_adapter(
        _make_handler(captured, add_sync_body={"success": False, "error": "bad"})
    )
    try:
        with caplog.at_level(logging.ERROR):
            out = asyncio.run(
                adapter.distill_round(
                    day="day01",
                    round_id="round-1",
                    round_index=1,
                    query="q",
                    answer="a",
                )
            )
    finally:
        adapter.close()
    assert out["ok"] is False
    assert any(EVOLVE_FAILURE_MARKER in rec.message for rec in caplog.records)


def test_paper_compat_surface_noop_evolve():
    captured: List[httpx.Request] = []
    adapter = _make_adapter(_make_handler(captured), evolve_every_n_rounds=3)
    try:
        assert adapter.expected_trigger_sessions == 3
        assert adapter.should_evolve([]) is True
        assert asyncio.run(adapter.evolve([], {})) == []
        assert hasattr(adapter, "update_history")
        assert hasattr(adapter, "history_path")
    finally:
        adapter.close()
