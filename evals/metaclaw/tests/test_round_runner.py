"""Tests for evals.metaclaw.round_runner."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evals.metaclaw.round_runner import (
    parse_bbox_answer,
    score_file_check,
    score_multi_choice,
    _tool_read_file,
    _tool_write_file,
    _tool_list_dir,
    _tool_bash,
)


def test_parse_bbox_simple():
    assert parse_bbox_answer("Final: \\bbox{A,E}") == ["A", "E"]


def test_parse_bbox_single_letter():
    assert parse_bbox_answer("\\bbox{B}") == ["B"]


def test_parse_bbox_with_spaces():
    assert parse_bbox_answer("answer \\bbox{ A , C , F } done") == ["A", "C", "F"]


def test_parse_bbox_missing_returns_empty():
    assert parse_bbox_answer("nope, no answer here") == []


def test_score_multi_choice_correct_set_equality_ignores_order():
    eval_block = {"answer": ["A", "E"]}
    assert score_multi_choice("\\bbox{E,A}", eval_block) == (1.0, "pass")


def test_score_multi_choice_wrong_subset():
    eval_block = {"answer": ["A", "E"]}
    assert score_multi_choice("\\bbox{A}", eval_block) == (0.0, "fail")


def test_score_multi_choice_wrong_extra():
    eval_block = {"answer": ["A", "E"]}
    assert score_multi_choice("\\bbox{A,E,F}", eval_block) == (0.0, "fail")


def test_score_file_check_passes_when_command_exits_0(tmp_path: Path):
    eval_block = {
        "command": "true",      # POSIX: exits 0
        "expect_exit": 0,
    }
    reward, outcome = score_file_check(eval_block, workspace=tmp_path)
    assert (reward, outcome) == (1.0, "pass")


def test_score_file_check_fails_when_command_exits_nonzero(tmp_path: Path):
    eval_block = {"command": "false", "expect_exit": 0}
    reward, outcome = score_file_check(eval_block, workspace=tmp_path)
    assert (reward, outcome) == (0.0, "fail")


def _make_fake_openai(responses: list[dict]):
    """Build a fake OpenAI client whose chat.completions.create returns
    the given responses in order. Each entry is either:
        {"text": str, "tool_calls": [{"id":..., "name":..., "args":{...}}]}
    """
    import json
    from types import SimpleNamespace

    class Calls:
        def __init__(self):
            self._idx = 0
        def create(self, **kwargs):
            r = responses[self._idx]
            self._idx += 1
            tcs = []
            for i, tc in enumerate(r.get("tool_calls", []) or []):
                tcs.append(SimpleNamespace(
                    id=tc.get("id", f"call-{i}"),
                    function=SimpleNamespace(
                        name=tc["name"],
                        arguments=json.dumps(tc.get("args", {})),
                    ),
                ))
            choice = SimpleNamespace(message=SimpleNamespace(
                content=r.get("text"), tool_calls=tcs or None,
            ))
            return SimpleNamespace(choices=[choice])

    chat = SimpleNamespace(completions=Calls())
    return SimpleNamespace(chat=chat)


def test_run_round_multi_choice_pass(tmp_path: Path):
    from evals.metaclaw.round_runner import RunnerConfig, run_round
    fake = _make_fake_openai([
        {"text": "Answer: \\bbox{A,E}"}                    # no tool calls → terminate
    ])
    res = run_round(
        openai_client=fake,
        cfg=RunnerConfig(chat_model="x", workspace=tmp_path, max_turns=3),
        round_id="r3", round_type="multi_choice",
        question="Q?", eval_block={"answer": ["A", "E"]},
        skills=[],
    )
    assert res.reward == 1.0
    assert res.eval_outcome == "pass"
    assert res.final_answer.endswith("\\bbox{A,E}")


def test_run_round_file_check_writes_then_passes(tmp_path: Path):
    from evals.metaclaw.round_runner import RunnerConfig, run_round
    fake = _make_fake_openai([
        # turn 0: tool call to write a file
        {"text": None, "tool_calls": [
            {"id": "c0", "name": "write_file",
             "args": {"path": "out.txt", "content": "hello"}},
        ]},
        # turn 1: no tool calls → terminate
        {"text": "Done."},
    ])
    res = run_round(
        openai_client=fake,
        cfg=RunnerConfig(chat_model="x", workspace=tmp_path, max_turns=3),
        round_id="r1", round_type="file_check",
        question="Write hello to out.txt",
        eval_block={"command": "test -f out.txt && grep -q hello out.txt",
                    "expect_exit": 0},
        skills=[],
    )
    assert res.reward == 1.0
    assert (tmp_path / "out.txt").read_text() == "hello"


def test_run_round_turn_limit_marks_error(tmp_path: Path):
    from evals.metaclaw.round_runner import RunnerConfig, run_round
    # Always emit one tool call → the loop never terminates naturally
    fake = _make_fake_openai([
        {"text": None, "tool_calls": [
            {"id": f"c{i}", "name": "list_dir", "args": {"path": "."}}
        ]} for i in range(10)
    ])
    res = run_round(
        openai_client=fake,
        cfg=RunnerConfig(chat_model="x", workspace=tmp_path, max_turns=2),
        round_id="r9", round_type="file_check",
        question="?",
        eval_block={"command": "false", "expect_exit": 0},
        skills=[],
    )
    assert res.error == "turn_limit"
    assert res.reward == 0.0


# -- Regression tests for code-review-driven fixes ---------------------------

def test_tool_read_file_rejects_traversal_via_sibling_prefix(tmp_path: Path):
    """`/tmp/ws_evil/secret.txt` must not be treated as inside `/tmp/ws/`.

    Regression for the substring-prefix check that previously accepted any
    path whose string-form started with the workspace path string.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    sibling = tmp_path / "ws_evil"
    sibling.mkdir()
    (sibling / "secret.txt").write_text("forbidden", encoding="utf-8")

    # Use a relative path that resolves into the sibling directory.
    out = _tool_read_file(workspace, "../ws_evil/secret.txt")
    assert "ERROR: path escapes workspace" in out


def test_tool_read_file_rejects_absolute_outside_path(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    elsewhere = tmp_path / "elsewhere.txt"
    elsewhere.write_text("nope", encoding="utf-8")

    out = _tool_read_file(workspace, str(elsewhere))
    assert "ERROR: path escapes workspace" in out


def test_tool_write_file_rejects_traversal(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()

    out = _tool_write_file(workspace, "../escaped.txt", "x")
    assert "ERROR: path escapes workspace" in out
    assert not (tmp_path / "escaped.txt").exists()


def test_tool_list_dir_rejects_traversal(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (tmp_path / "outside").mkdir()

    out = _tool_list_dir(workspace, "../outside")
    assert "ERROR: path escapes workspace" in out


def test_tool_bash_timeout_returns_message_not_exception(tmp_path: Path, monkeypatch):
    """A bash command that exceeds BASH_TIMEOUT_S must not propagate
    `subprocess.TimeoutExpired`; instead it returns a string with exit=124.
    """
    import evals.metaclaw.round_runner as rr
    # Reduce timeout so the test is fast
    monkeypatch.setattr(rr, "BASH_TIMEOUT_S", 1)

    out = rr._tool_bash(tmp_path, "sleep 5")
    assert "exit=124" in out
    assert "timeout" in out.lower()


def test_score_file_check_timeout_returns_fail(tmp_path: Path, monkeypatch):
    import evals.metaclaw.round_runner as rr
    monkeypatch.setattr(rr, "SCORE_FILE_CHECK_TIMEOUT_S", 1)

    reward, outcome = rr.score_file_check(
        {"command": "sleep 5", "expect_exit": 0}, workspace=tmp_path
    )
    assert (reward, outcome) == (0.0, "fail")
