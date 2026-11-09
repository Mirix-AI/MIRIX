import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "evals"))
    return (
        importlib.import_module("main_eval"),
        importlib.import_module("task_agent"),
        importlib.import_module("mirix_memory_system"),
    )


def _run_eval(monkeypatch, tmp_path, main, *, qa_only, window):
    conversation = {
        "session_1": [{"speaker": "A", "text": f"Turn {n}"} for n in range(5)],
        "session_1_date_time": "2023-01-01T12:00:00",
        "session_4": [{"speaker": "A", "text": "Later session"}],
        "session_4_date_time": "2023-04-01T12:00:00",
    }
    data = tmp_path / "data.json"
    data.write_text(json.dumps([{"sample_id": "sample", "conversation": conversation, "qa": []}]))
    task = MagicMock(return_value=SimpleNamespace(mirix_client=object()))
    memory = SimpleNamespace(add_chunk=MagicMock(return_value={}), list_all_memories=lambda: {})
    monkeypatch.setattr(main, "TaskAgent", task)
    monkeypatch.setattr(main, "MirixMemorySystem", MagicMock(return_value=memory))
    monkeypatch.setattr(main, "extract_selective_visual_evidence", lambda *a: "")
    monkeypatch.setattr(main, "is_v720", lambda _: True)
    monkeypatch.setenv("MIRIX_WINDOW_TURNS", str(window))
    monkeypatch.setenv("MIRIX_QA_ONLY_EXISTING_STORE", "1" if qa_only else "0")
    monkeypatch.setenv("MIRIX_DREAM_EVERY_N_CHUNKS", "3" if qa_only else "0")
    monkeypatch.setenv("MIRIX_EVAL_USER_PREFIX", "")
    post = MagicMock(return_value=SimpleNamespace(json=lambda: {"stats": {}}))
    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(httpx, "get", MagicMock(return_value=SimpleNamespace(json=lambda: {"stats": {}})))
    monkeypatch.setattr(sys, "argv", [
        "main_eval.py", "--data", str(data), "--output_path", str(tmp_path / "out"),
        "--mirix_config_path", str(tmp_path / "unused.yaml"),
    ])
    main.main()
    return task, memory, post


def test_windowed_eval_uses_the_original_sessions_date(modules, monkeypatch, tmp_path):
    _, memory, _ = _run_eval(monkeypatch, tmp_path, modules[0], qa_only=False, window=2)
    calls = memory.add_chunk.call_args_list
    assert len(calls) == 5
    for call, date, number in zip(
        calls,
        ["2023-01-01"] * 4 + ["2023-04-01"],
        [1, 1, 1, 1, 4],
    ):
        assert f"timestamped at {date}T12:00:00" in call.args[0]
        assert call.kwargs["source_meta"]["mentioned_at"].startswith(date)
        assert call.kwargs["source_meta"]["session_id"] == number


def test_qa_only_skips_tail_dream_and_initialization(modules, monkeypatch, tmp_path):
    task, memory, post = _run_eval(monkeypatch, tmp_path, modules[0], qa_only=True, window=0)
    assert task.call_args.kwargs["read_only"] is True
    memory.add_chunk.assert_not_called()
    assert not any("/memory/auto_dream" in call.args[0] for call in post.call_args_list)


def test_qa_only_task_attaches_existing_agent_without_initialization(modules, monkeypatch):
    task = modules[1]
    from mirix.schemas.agent import AgentType

    agent = SimpleNamespace(agent_type=AgentType.meta_memory_agent)
    client = SimpleNamespace(
        list_agents=AsyncMock(return_value=[agent]),
        initialize_meta_agent=AsyncMock(),
        _client=SimpleNamespace(event_hooks={"request": []}),
    )
    monkeypatch.setattr(task, "MirixClient", lambda **kw: client)
    monkeypatch.setattr(task, "OpenAI", lambda **kw: object())
    task.TaskAgent("/nonexistent/config.yaml", api_key="test-only", read_only=True)
    assert client._meta_agent is agent
    client.initialize_meta_agent.assert_not_awaited()
    assert client._client.event_hooks["request"]


def test_qa_only_does_not_create_a_missing_agent(modules, monkeypatch):
    task = modules[1]
    client = SimpleNamespace(list_agents=AsyncMock(return_value=[]), initialize_meta_agent=AsyncMock())
    monkeypatch.setattr(task, "MirixClient", lambda **kw: client)
    monkeypatch.setattr(task, "OpenAI", lambda **kw: object())
    with pytest.raises(ValueError, match="existing meta agent"):
        task.TaskAgent("/nonexistent/config.yaml", api_key="test-only", read_only=True)
    client.initialize_meta_agent.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(("method", "path"), [
    ("POST", "/memory/add"),
    ("POST", "/memory/auto_dream"),
    ("POST", "/agents/meta/initialize"),
    ("DELETE", "/memory/semantic/test"),
    ("PATCH", "/clients/test"),
])
async def test_qa_only_network_guard_rejects_mutation(modules, method, path):
    request = httpx.Request(method, "http://test" + path)
    with pytest.raises(RuntimeError, match="QA-only"):
        await modules[1].TaskAgent._guard_read_only_request(request)


def test_qa_only_memory_system_rejects_ingest(modules):
    client = SimpleNamespace(add=AsyncMock())
    memory = modules[2].MirixMemorySystem(client=client, read_only=True)
    with pytest.raises(RuntimeError, match="QA-only"):
        memory.add_chunk("This must never be ingested")
    client.add.assert_not_called()
