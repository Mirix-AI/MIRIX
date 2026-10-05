"""Shared helpers for the baseline tests (ReasoningBank, AWM): offline, fake upstream only.

:class:`Router` answers the fake upstream's calls by who made them: J (its frozen system
prompt), the baselines' learner (RB's Figure 9 instructions, AWM's induction prompt), the
embedding endpoint (scripted vectors) and the actor (a :class:`rollout_support.Script`).
"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from evals.skill_bench import protocol
from evals.skill_bench.methods import awm as A
from evals.skill_bench.methods import judge as J
from evals.skill_bench.methods import reasoningbank as R
from evals.skill_bench.tests.fakes import Reply, embedding_body
from evals.skill_bench.tests.rollout_support import system_text, text_reply


def judged(status: str, thoughts: str = "the agent's steps reach the goal") -> str:
    """A well-formed J reply."""
    return f"Thoughts: {thoughts}\nStatus: {status}"


def items_reply(*titles: str, description: bool = True) -> str:
    """An extraction reply in Figure 9's format, one item per title."""
    blocks = []
    for i, title in enumerate(titles, start=1):
        lines = [f"# Memory Item {i}", f"## Title {title}"]
        if description:
            lines.append(f"## Description DESC-{title}: when it applies")
        lines.append(f"## Content content of {title}.")
        blocks.append("\n".join(lines))
    return "```\n" + "\n\n".join(blocks) + "\n```"


def first_text(body: Mapping[str, Any]) -> str:
    """The text of the last message of a request (the first text part of a list)."""
    content = body["messages"][-1].get("content")
    if isinstance(content, list):
        return next((p["text"] for p in content if p.get("type") == "text"), "")
    return content if isinstance(content, str) else ""


def call_kind(body: Mapping[str, Any]) -> str:
    """``judge``, ``learner`` or ``actor``: who sent a chat request."""
    system = system_text(body)
    if system == J.system_prompt():
        return "judge"
    if system in (R.system_prompt("success"), R.system_prompt("failure")):
        return "learner"
    if len(body["messages"]) == 1 and first_text(body).startswith(A.instruction_text()):
        return "learner"
    return "actor"


class Router:
    """FakeUpstream responder for the baselines' tests.

    ``judge`` / ``learner`` are consumed in order (a string becomes a stop reply); when a
    queue is empty the call gets the fake's generic reply. ``vectors`` maps an embedding
    input to its vector (others get the fake's hash vector). ``actor`` answers actor calls.
    Every request is kept in ``calls`` by kind.
    """

    def __init__(
        self,
        *,
        judge: Iterable[str | Reply] = (),
        learner: Iterable[str | Reply] = (),
        vectors: Mapping[str, Sequence[float]] | None = None,
        actor: Callable[[str, dict[str, Any]], Reply | None] | None = None,
    ) -> None:
        self.judge = list(judge)
        self.learner = list(learner)
        self.vectors = {k: list(v) for k, v in (vectors or {}).items()}
        self.actor = actor
        self.calls: dict[str, list[dict[str, Any]]] = {
            "judge": [],
            "learner": [],
            "embeddings": [],
            "actor": [],
        }

    def __call__(self, endpoint: str, body: dict[str, Any]) -> Reply | None:
        if endpoint == "embeddings":
            self.calls["embeddings"].append(body)
            inputs = (
                body["input"] if isinstance(body["input"], list) else [body["input"]]
            )
            data = embedding_body([str(i) for i in inputs])
            for row, text in zip(data["data"], inputs):
                if text in self.vectors:
                    row["embedding"] = self.vectors[text]
            return Reply(json=data)
        kind = call_kind(body)
        self.calls[kind].append(body)
        if kind == "actor":
            return self.actor(endpoint, body) if self.actor else None
        queue = self.judge if kind == "judge" else self.learner
        if not queue:
            return None
        reply = queue.pop(0)
        return reply if isinstance(reply, Reply) else text_reply(reply)

    def embedded(self) -> list[list[str]]:
        return [list(b["input"]) for b in self.calls["embeddings"]]


def ledger(tmp_path: Path, run_id: str) -> list[dict[str, Any]]:
    path = tmp_path / "runs" / run_id / "gateway.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [row for row in rows if row.get("event") == "call"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normal(text: str) -> str:
    """Whitespace-insensitive form."""
    return " ".join(text.split())


# --- reading the verbatim originals -----------------------------------------------------------


def module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def string_constant(tree: ast.Module, name: str) -> str:
    """A module-level string assignment ``NAME = "..."``."""
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    raise AssertionError(f"{name} not found")


def function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def fstring_template(node: ast.JoinedStr, names: Mapping[str, str]) -> str:
    """An f-string with each expression replaced by ``names[source]`` (e.g. ``{query}``)."""
    out = []
    for part in node.values:
        if isinstance(part, ast.Constant):
            out.append(str(part.value))
        else:
            assert isinstance(part, ast.FormattedValue)
            out.append(names[ast.unparse(part.value)])
    return "".join(out)


def originals_match(prompt_dir: Path, provenance: Mapping[str, Any]) -> list[str]:
    """Checked files: each recorded sha256 (and git blob id) matches the stored copy."""
    import hashlib

    checked = []
    for rel, entry in provenance.items():
        data = (prompt_dir / rel).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], rel
        if "git_blob" in entry:
            blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
            assert blob == entry["git_blob"], rel
        assert protocol.is_fingerprinted(
            Path(prompt_dir.relative_to(protocol.SKILL_BENCH_DIR)) / rel
        ), rel
        checked.append(rel)
    return checked
