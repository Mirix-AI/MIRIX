"""Shared helpers for the phase-11 actor/rollout/trajectory tests (offline, fake upstream only)."""

from __future__ import annotations

import copy
import json
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx

from evals.skill_bench import protocol
from evals.skill_bench.agent.actor import Actor
from evals.skill_bench.envs.base import Message
from evals.skill_bench.infra.gateway import Gateway
from evals.skill_bench.methods.base import (
    Injection,
    MemoryMethodBase,
    TaskView,
    TrajectoryView,
)
from evals.skill_bench.tests.fakes import (
    FakeUpstream,
    Reply,
    chat_body,
    running_gateway,
)

LUNA = protocol.MODELS["luna"]
RUN_ID = "cell-test-a1"
SALT = b"phase-11-test-salt-0123456789abcdef"

# ---------------------------------------------------------------------------
# Gateway + actor
# ---------------------------------------------------------------------------


@asynccontextmanager
async def gateway_actor(
    tmp_path: Path,
    fake: FakeUpstream,
    *,
    model: str = LUNA,
    run_id: str = RUN_ID,
    **gateway_config: Any,
) -> AsyncIterator[tuple[Gateway, Actor]]:
    """A real in-process gateway over a fake upstream, and an actor talking to it over ASGI."""
    async with running_gateway(tmp_path, fake, **gateway_config) as (gateway, _client):
        actor = Actor(
            gateway_base_url="http://gateway",
            run_id=run_id,
            model=model,
            transport=httpx.ASGITransport(app=gateway.app),
        )
        async with actor:
            yield gateway, actor


def ledger_calls(tmp_path: Path, run_id: str = RUN_ID) -> list[dict[str, Any]]:
    path = tmp_path / "runs" / run_id / "gateway.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [row for row in rows if row.get("event") == "call"]


# ---------------------------------------------------------------------------
# Scripted model
# ---------------------------------------------------------------------------


def text_reply(
    content: str | None,
    *,
    finish_reason: str = "stop",
    reasoning: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    model: str = LUNA,
) -> Reply:
    return Reply(
        json=chat_body(
            model=model,
            content=content,
            finish_reason=finish_reason,
            reasoning=reasoning,
            tool_calls=tool_calls,
        )
    )


def system_text(body: Mapping[str, Any]) -> str:
    messages = body.get("messages") or []
    if messages and messages[0].get("role") == "system":
        content = messages[0].get("content")
        return content if isinstance(content, str) else ""
    return ""


def benchmark_of(body: Mapping[str, Any]) -> str:
    """Which benchmark an actor request belongs to, from its frozen system prompt (WebShop,
    which has none: from its frozen user prompt)."""
    system = system_text(body)
    if system.startswith("You are an expert agent operating in the ALFRED"):
        return "alfworld"
    messages = body.get("messages") or []
    if (
        not system
        and messages
        and WEBSHOP_ACTOR_MARK in str(messages[0].get("content") or "")
    ):
        # verl-agent's prompt, no system message: the conversation's first user message.
        return "webshop"
    return "unknown"


#: A phrase of WebShop's first actor message (``envs/templates/webshop``).
WEBSHOP_ACTOR_MARK = "operating in the WebShop e\u2011commerce environment"


class Script:
    """Per-benchmark reply lists, consumed in order; used as a FakeUpstream responder."""

    def __init__(self, **replies: Sequence[Reply]) -> None:
        self.pending = {k: list(v) for k, v in replies.items()}
        self.seen: dict[str, list[dict[str, Any]]] = {k: [] for k in replies}

    def __call__(self, endpoint: str, body: dict[str, Any]) -> Reply | None:
        if endpoint != "chat/completions":
            return None
        bench = benchmark_of(body)
        self.seen.setdefault(bench, []).append(body)
        queue = self.pending.get(bench)
        if not queue:
            raise AssertionError(f"no scripted reply left for {bench}")
        return queue.pop(0)


# ---------------------------------------------------------------------------
# Recording memory method
# ---------------------------------------------------------------------------


class RecordingMethod(MemoryMethodBase):
    """Records every hook call; injects ``memory`` (or what ``inject`` returns)."""

    name = "recording"

    def __init__(
        self,
        memory: str = "",
        *,
        item_ids: Sequence[str] = (),
        fail: Mapping[str, BaseException] | None = None,
        inject: Callable[[TaskView], Any] | None = None,
    ) -> None:
        super().__init__()
        self.memory = memory
        self.item_ids = tuple(item_ids)
        self.fail = dict(fail or {})
        self.inject = inject
        self.events: list[tuple[str, Any]] = []

    def _maybe_fail(self, hook: str) -> None:
        if hook in self.fail:
            raise self.fail[hook]

    async def before_task(self, task: TaskView) -> Injection:
        self.events.append(("before_task", task))
        self._maybe_fail("before_task")
        if self.inject is not None:
            return self.inject(task)
        return Injection(self.memory, self.item_ids, tokens=999_999)  # ignored

    async def on_message(self, task_uid: str, message: Message) -> None:
        self.events.append(("on_message", (task_uid, copy.deepcopy(message))))
        self._maybe_fail("on_message")
        message["content"] = "MUTATED BY METHOD"  # must not affect the rollout

    async def after_task(self, trajectory: TrajectoryView) -> None:
        self.events.append(("after_task", trajectory))
        self._maybe_fail("after_task")

    async def wait_until_idle(self, timeout_s: float) -> None:
        self.events.append(("wait_until_idle", timeout_s))
        self._maybe_fail("wait_until_idle")

    def hooks(self) -> list[str]:
        return [name for name, _ in self.events]

    def received(self) -> list[Any]:
        """Everything the method was given (for label scans)."""
        return [payload for _, payload in self.events]


# ---------------------------------------------------------------------------
# "Only the memory slot differs"
# ---------------------------------------------------------------------------


def inserted_text(plain: str, with_memory: str) -> str:
    """The single insertion that turns ``plain`` into ``with_memory``; fails otherwise."""
    prefix = 0
    limit = min(len(plain), len(with_memory))
    while prefix < limit and plain[prefix] == with_memory[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < len(plain) - prefix
        and suffix < len(with_memory) - prefix
        and plain[-1 - suffix] == with_memory[-1 - suffix]
    ):
        suffix += 1
    assert prefix + suffix == len(plain), "the plain prompt is not a sub-sequence"
    return with_memory[prefix : len(with_memory) - suffix]


def slot_insertions(plain: Any, with_memory: Any, path: str = "$") -> list[str]:
    """Walk two request bodies; every difference must be one insertion into one string.

    Returns the inserted texts (one per differing string). Keys, list lengths and all
    non-string values must be identical.
    """
    if isinstance(plain, Mapping):
        assert isinstance(with_memory, Mapping) and list(plain) == list(with_memory), (
            path
        )
        return [
            text
            for key in plain
            for text in slot_insertions(plain[key], with_memory[key], f"{path}.{key}")
        ]
    if isinstance(plain, list):
        assert isinstance(with_memory, list) and len(plain) == len(with_memory), path
        return [
            text
            for i, (a, b) in enumerate(zip(plain, with_memory))
            for text in slot_insertions(a, b, f"{path}[{i}]")
        ]
    if isinstance(plain, str) and isinstance(with_memory, str) and plain != with_memory:
        return [inserted_text(plain, with_memory)]
    assert plain == with_memory, path
    return []
