"""Single-round agent loop and scoring for the MetaClaw bench.

Tools follow the OpenAI function-calling schema:
  - bash(command):       run shell command in the round's workspace
  - read_file(path):     read text file (UTF-8, max 100KB)
  - write_file(path, content):  overwrite file with content
  - list_dir(path):      list directory entries

The loop terminates when:
  - the assistant message has no tool_calls (final answer reached), or
  - max_turns (default 20) is hit, or
  - wallclock cap (default 300 s) is hit.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from evals.metaclaw.format_adapter import RoundResult


_BBOX_RE = re.compile(r"\\bbox\{([^}]+)\}")


def parse_bbox_answer(text: str) -> list[str]:
    """Extract letters from the LAST occurrence of \\bbox{...}."""
    matches = _BBOX_RE.findall(text or "")
    if not matches:
        return []
    inner = matches[-1]
    letters = [s.strip().upper() for s in inner.split(",") if s.strip()]
    return letters


def score_multi_choice(final_answer: str, eval_block: dict) -> tuple[float, str]:
    expected = set(eval_block.get("answer", []))
    got = set(parse_bbox_answer(final_answer))
    return (1.0, "pass") if got == expected else (0.0, "fail")


def score_file_check(eval_block: dict, workspace: Path) -> tuple[float, str]:
    cmd = eval_block.get("command", "")
    expect = int(eval_block.get("expect_exit", 0))
    if not cmd:
        return (0.0, "fail")
    proc = subprocess.run(
        cmd, shell=True, cwd=str(workspace),
        capture_output=True, text=True, timeout=60,
    )
    return (1.0, "pass") if proc.returncode == expect else (0.0, "fail")


# -- Tools -------------------------------------------------------------------

def _tool_bash(workspace: Path, command: str) -> str:
    proc = subprocess.run(
        command, shell=True, cwd=str(workspace),
        capture_output=True, text=True, timeout=60,
    )
    out = (proc.stdout or "")[-4000:]
    err = (proc.stderr or "")[-2000:]
    return f"exit={proc.returncode}\nstdout:\n{out}\nstderr:\n{err}"


def _tool_read_file(workspace: Path, path: str) -> str:
    p = (workspace / path).resolve()
    if not str(p).startswith(str(workspace.resolve())):
        return "ERROR: path escapes workspace"
    if not p.exists():
        return f"ERROR: not found: {path}"
    data = p.read_bytes()[:102_400]
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def _tool_write_file(workspace: Path, path: str, content: str) -> str:
    p = (workspace / path).resolve()
    if not str(p).startswith(str(workspace.resolve())):
        return "ERROR: path escapes workspace"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} bytes to {path}"


def _tool_list_dir(workspace: Path, path: str = ".") -> str:
    p = (workspace / path).resolve()
    if not str(p).startswith(str(workspace.resolve())):
        return "ERROR: path escapes workspace"
    if not p.is_dir():
        return f"ERROR: not a directory: {path}"
    items = []
    for entry in sorted(p.iterdir()):
        kind = "d" if entry.is_dir() else "f"
        items.append(f"{kind}\t{entry.relative_to(workspace)}")
    return "\n".join(items) or "(empty)"


_TOOLS_SCHEMA: list[dict] = [
    {"type": "function", "function": {
        "name": "bash",
        "description": "Run a shell command in the workspace.",
        "parameters": {
            "type": "object", "additionalProperties": False,
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file relative to the workspace.",
        "parameters": {
            "type": "object", "additionalProperties": False,
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write (overwrite) a text file relative to the workspace.",
        "parameters": {
            "type": "object", "additionalProperties": False,
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        }}},
    {"type": "function", "function": {
        "name": "list_dir",
        "description": "List entries in a directory relative to the workspace.",
        "parameters": {
            "type": "object", "additionalProperties": False,
            "properties": {"path": {"type": "string", "default": "."}},
            "required": [],
        }}},
]


def _dispatch_tool(name: str, args: dict, workspace: Path) -> str:
    if name == "bash":
        return _tool_bash(workspace, args.get("command", ""))
    if name == "read_file":
        return _tool_read_file(workspace, args.get("path", ""))
    if name == "write_file":
        return _tool_write_file(workspace, args.get("path", ""), args.get("content", ""))
    if name == "list_dir":
        return _tool_list_dir(workspace, args.get("path", "."))
    return f"ERROR: unknown tool {name}"


# -- Loop --------------------------------------------------------------------

@dataclass
class RunnerConfig:
    chat_model: str
    workspace: Path
    max_turns: int = 20
    wallclock_cap_s: float = 300.0


SYSTEM_PROMPT_BASE = (
    "You are an agent solving a single task. The user will give you ONE "
    "question. Use the provided tools (bash, read_file, write_file, "
    "list_dir) to inspect the workspace and produce the requested output. "
    "When the task is complete, reply with a brief final message and STOP "
    "calling tools. For multiple-choice questions, end your final message "
    "with \\bbox{X} or \\bbox{X,Y}."
)


def build_system_prompt(skills: list[dict]) -> str:
    if not skills:
        return SYSTEM_PROMPT_BASE
    parts = [SYSTEM_PROMPT_BASE, "", "## Relevant skills"]
    for s in skills:
        parts.append(f"### {s['name']}  ({s.get('category','general')})")
        parts.append(s.get("description", "").strip())
        parts.append("")
        parts.append(s.get("content", "").strip())
        parts.append("")
    return "\n".join(parts)


def run_round(
    *,
    openai_client,
    cfg: RunnerConfig,
    round_id: str,
    round_type: str,
    question: str,
    eval_block: dict,
    skills: list[dict],
) -> RoundResult:
    system = build_system_prompt(skills)
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    transcript: list[dict] = []
    started = time.monotonic()
    final_text = ""
    error: str | None = None

    for turn in range(cfg.max_turns):
        if time.monotonic() - started > cfg.wallclock_cap_s:
            error = "wallclock_cap"
            break
        resp = openai_client.chat.completions.create(
            model=cfg.chat_model,
            messages=messages,
            tools=_TOOLS_SCHEMA,
            tool_choice="auto",
        )
        choice = resp.choices[0]
        msg = choice.message
        transcript.append({"role": "assistant", "content": msg.content,
                           "tool_calls": [
                               {"name": tc.function.name,
                                "arguments": tc.function.arguments}
                               for tc in (msg.tool_calls or [])
                           ]})
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in (msg.tool_calls or [])
            ],
        })
        if not msg.tool_calls:
            final_text = msg.content or ""
            break
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = _dispatch_tool(tc.function.name, args, cfg.workspace)
            transcript.append({"role": "tool", "name": tc.function.name,
                               "result": result[:1000]})
            messages.append({
                "role": "tool", "tool_call_id": tc.id, "content": result,
            })
    else:
        error = "turn_limit"

    # Score
    if round_type == "multi_choice":
        reward, outcome = score_multi_choice(final_text, eval_block)
    elif round_type == "file_check":
        reward, outcome = score_file_check(eval_block, cfg.workspace)
    else:
        reward, outcome = (0.0, "fail")
        error = error or f"unknown_round_type:{round_type}"

    return RoundResult(
        round_id=round_id, round_type=round_type, question=question,
        final_answer=final_text, reward=reward, eval_outcome=outcome,
        feedback="",   # filled in by driver from questions.json
        transcript=transcript, error=error,
    )
