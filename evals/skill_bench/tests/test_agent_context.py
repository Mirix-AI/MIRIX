"""The agent context: the executor's prompt template reaches memory
methods as ``TaskView.agent_context``, a static, label-free text built from the very
templates the actor's prompts are rendered from (``envs/agent_context.py``).

* rendering each section of an environment's agent context with the step's values is the
  actor's real prompt, on every step of real (scripted) ALFWorld and WebShop episodes,
  with and without memory; WebShop's sections, laid out as its conversation grows (the
  first user message, then each step's reply and the next step's user message), are its
  whole request, a call without a reply included;
* the text holds every section verbatim and nothing but the described placeholders;
* it is the same for every item and holds no label of any item of any split;
* the rollout hands it to methods in both phases.
"""

from __future__ import annotations

import asyncio
import hashlib
import string
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from evals.skill_bench import audit, protocol
from evals.skill_bench.agent import rollout as ro
from evals.skill_bench.envs import agent_context as ac
from evals.skill_bench.envs import alfworld as alf
from evals.skill_bench.envs import webshop as ws
from evals.skill_bench.envs.base import Task
from evals.skill_bench.methods.base import TaskView
from evals.skill_bench.tests.fakes import FakeUpstream
from evals.skill_bench.tests.rollout_support import (
    SALT,
    RecordingMethod,
    gateway_actor,
    text_reply,
)
from evals.skill_bench.tests.test_env_alfworld import Games, act, make_env
from evals.skill_bench.tests.webshop_support import make_shop, respond, shopper_action

#: A memory text with braces, quotes and lines: it must go in verbatim.
MEMORY = "## Skills\n- check the {receptacle} first & 'quote' \"x\"\n- then act"


def _section_for(sections: Sequence[ac.PromptSection], role: str, step: int) -> str:
    (section,) = [s for s in sections if s.role == role and s.sent_at(step)]
    return section.template


def _fill(template: str, memory_slot: str, values: Mapping[str, Any]) -> str:
    """A section filled with the step's values; without memory the slot and its blank
    line are left out (what the agent context says, ``insert_memory`` does)."""
    if not str(values["memory"]).strip():
        assert template.startswith(memory_slot)
        template = template[len(memory_slot) :]
    return template.format(**values)


def _alfworld_values(
    t: alf.AlfWorldTemplates, fields: Mapping[str, Any], memory: str
) -> dict[str, Any]:
    history = fields["history"]
    return {
        "memory": memory,
        "current_observation": fields["observation"],
        "admissible_actions": alf.format_admissible_actions(
            t, fields["admissible_actions"]
        ),
        "task_description": fields["task_description"],
        "step_count": len(history),
        "current_step": len(history) + 1,
        "action_history": alf.format_action_history(t, history)[0],
    }


def _alfworld_rendered(
    t: alf.AlfWorldTemplates, fields: Mapping[str, Any], memory: str, step: int
) -> list[dict[str, str]]:
    sections = alf.agent_context_sections(t)
    slot = t.memory_slot + t.memory_joiner
    values = _alfworld_values(t, fields, memory)
    return [
        {"role": "system", "content": _section_for(sections, "system", step)},
        {
            "role": "user",
            "content": _fill(_section_for(sections, "user", step), slot, values),
        },
    ]


def _webshop_rendered(
    t: ws.WebShopTemplates,
    seen: Sequence[Mapping[str, Any]],
    replies: Sequence[str],
    memory: str,
) -> list[dict[str, str]]:
    """The request of step ``len(seen)`` as the agent context lays it out: the first
    user message with step 1's values, then for each earlier step its reply and the user
    message of the step after it. ``seen`` holds each step's fields, ``replies`` each
    reply as the agent context describes it (the placeholder for none)."""
    first, said, turn = ws.agent_context_sections(t)
    slot = t.memory_slot + t.memory_joiner

    def values(fields: Mapping[str, Any], step: int) -> dict[str, Any]:
        return {
            "memory": memory,
            "task_description": fields["task_description"],
            "current_observation": fields["observation"],
            "available_actions": ws.format_available_actions(
                t, fields["available_actions"]
            ),
            "current_step": step,
        }

    messages = [
        {"role": first.role, "content": _fill(first.template, slot, values(seen[0], 1))}
    ]
    for step, fields in enumerate(seen[1:], start=2):
        reply = said.template.format(reply=replies[step - 2])
        messages += [
            {"role": said.role, "content": reply},
            {
                "role": turn.role,
                "content": turn.template.format(**values(fields, step)),
            },
        ]
    return messages


# --- the text is exactly what render_actor_messages fills ---------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("memory", ["", MEMORY, " \n\t"])
async def test_alfworld_sections_filled_in_are_the_actor_prompts(tmp_path, memory):
    """Every step of a scripted ALFWorld episode (step 1, the filling history of step 2,
    the full history after it): each section, filled with that step's values, is the
    prompt the actor is sent. The observations carry braces, which stay literal."""
    env = make_env(Games(), tmp_path)
    task = env.load_tasks("train")[0]
    t = env.templates
    checked = 0
    async with env.episode(task) as episode:
        fields = (await episode.reset()).fields
        for step, action in enumerate(
            ["go to countertop 1", "look", "inventory", "look", "win"], start=1
        ):
            assert len(fields["history"]) == step - 1
            assert _alfworld_rendered(t, fields, memory, step) == env.actor_messages(
                fields, memory
            )
            checked += 1
            result = await episode.step(act(action))
            fields = result.fields
    assert checked == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("memory", ["", MEMORY])
async def test_webshop_sections_filled_in_are_the_actor_prompts(tmp_path, memory):
    """Every step of WebShop episodes on the scripted shop (the real server.py): the
    sections, filled with each step's values and laid out as the conversation grows,
    are the request the actor is sent; a call without a reply and
    an empty reply show the placeholder the agent context names."""
    shop = make_shop(tmp_path)
    env = shop.env()
    t = env.templates
    checked = set()
    for index, right in ((0, True), (1, False)):
        task = env.load_tasks("test")[index]
        async with env.episode(task) as episode:
            fields = (await episode.reset()).fields
            seen: list[Mapping[str, Any]] = []
            replies: list[str] = []
            for step in range(1, protocol.WEBSHOP_MAX_STEPS + 1):
                seen.append(fields)
                expected = env.actor_messages(fields, memory)
                assert _webshop_rendered(t, seen, replies, memory) == expected
                checked.add(min(step, 3))
                if not right and step == 2:
                    action: Any = env.call_error_response()
                    replies.append(t.no_reply_turn)
                elif not right and step == 3:
                    action = " \n"
                    replies.append(t.no_reply_turn)
                else:
                    action = respond(shopper_action(expected, right=right))
                    replies.append(action)
                result = await episode.step(action)
                fields = result.fields
                if result.done:
                    break
    assert checked == {1, 2, 3}  # the first message, one exchange, several


def test_webshop_text_describes_the_conversation():
    """WebShop's agent context says the prompt is one conversation
    that grows by one exchange per step and keeps every earlier step; its sections are
    the first user message (behind the memory slot), the reply and the later user
    message, each verbatim; no 13,000-character fallback, no window."""
    t = ws.WebShopTemplates.load()
    text = ws.agent_context(t)
    assert text.startswith(ws.CONVERSATION_INTRO + "\n- {memory}: ")
    assert ac.INTRO not in text
    assert "at step N the agent sees all N-1 earlier steps" in text
    assert [line for line in text.splitlines() if line.startswith(ac.SECTION_MARK)] == [
        "=== user message, step 1 ===",
        "=== assistant message, the agent's reply at each step ===",
        "=== user message, appended at step 2 and later ===",
    ]
    first, said, turn = ws.agent_context_sections(t)
    assert first.template == "{memory}\n\n" + t.no_history
    assert said.template == "{reply}" and turn.template == t.turn
    for section in (first, said, turn):
        assert f"{section.title} {ac.SECTION_MARK}\n{section.template}" in text
    names = [n for s in (first, said, turn) for n in ac.fields_of(s.template)]
    for name in names:
        assert f"- {{{name}}}: " in text
    head = text.split(f"\n\n{ac.SECTION_MARK} ", 1)[0]
    assert f"shows {t.no_reply_turn}" in head and "in the first message only" in head
    assert "cannot be sent shows as U+FFFD" in head  # reply_turn's without_surrogates
    # Nothing in the text is a format field the prompts do not have.
    assert {n for _, n, _, _ in string.Formatter().parse(head) if n} <= set(names)
    for gone in ("13,000", "13000", "{action_history}", "{step_count}", "most recent"):
        assert gone not in text


def test_the_text_holds_every_section_verbatim_and_describes_every_slot():
    """ALFWorld's (WebShop's conversation: test_webshop_text_describes_the_conversation)."""
    t = alf.AlfWorldTemplates.load()
    text = alf.agent_context(t)
    sections = alf.agent_context_sections(t)
    for section in sections:
        heading = f"{ac.SECTION_MARK} {section.title} {ac.SECTION_MARK}\n"
        assert heading + section.template in text
    assert text.count(f"\n{ac.SECTION_MARK} ") == len(sections)
    # Sections cover every step exactly once, per role.
    for role in {s.role for s in sections}:
        for step in range(1, 60):
            assert len([s for s in sections if s.role == role and s.sent_at(step)]) == 1
    # The history length is filled in (it is no placeholder); the per-step slots are.
    names = {n for s in sections for n in ac.fields_of(s.template)}
    assert "history_length" not in names
    assert {"memory", "current_observation", "step_count", "action_history"} <= names
    for name in names:
        assert f"- {{{name}}}: " in text
    assert "{memory}\n\n" in text  # the Q21 slot heads every user message
    head = text.split(f"\n\n{ac.SECTION_MARK} ", 1)[0]
    assert "{history_length}" not in text
    assert f"the {t.history_length} most recent earlier steps" in head
    # Nothing in the text is a format field the prompts do not have.
    assert {
        n for _, n, _, _ in string.Formatter().parse(head.replace("{N}", "")) if n
    } <= names


def test_alfworld_text_head():
    """What ALFWorld's agent context starts with, and its three later-step parts."""
    text = alf.agent_context(alf.AlfWorldTemplates.load())
    assert text.startswith(ac.INTRO + "\n- {memory}: " + ac.MEMORY_DESCRIPTION)
    assert [line for line in text.splitlines() if line.startswith(ac.SECTION_MARK)] == [
        "=== system message, every step ===",
        "=== user message, step 1 ===",
        "=== user message, step 2 ===",
        "=== user message, step 3 and later ===",
    ]
    assert "Below are the most recent 1 observations" in text
    assert "Below are the most recent 2 observations" in text


def test_webshop_text_never_states_the_step_cap():
    """D5: the prompt never states the 15-step cap, so neither does its agent context."""
    text = ws.agent_context(ws.WebShopTemplates.load())
    assert str(protocol.WEBSHOP_MAX_STEPS) not in text
    assert "system message" not in text  # verl-agent sends no system message


# --- static and label-free ---------------------------------------------------------------


def _all_tasks(env: Any, benchmark: str) -> list[Task]:
    splits = ("train", "val", "test")
    return [task for split in splits for task in env.load_tasks(split)]


@pytest.mark.parametrize("benchmark", ["alfworld", "webshop"])
def test_the_agent_context_is_the_same_for_every_item_and_holds_no_label(benchmark):
    """The audit's H10.1 search over the agent context of every item of every split
    (``env.label_strings``, the raw id, the gold answers and the category labels): none
    occurs. The text depends on the templates alone, so it is one text per benchmark."""

    def make() -> Any:
        if benchmark == "alfworld":
            return alf.AlfWorldEnv(backend_factory=Games().factory())
        return ws.WebShopEnv()

    env = make()
    text = env.agent_context()
    assert text and text == make().agent_context()
    corpus = audit.Corpus()
    corpus.add(audit.normalise(text), "agent_context", "agent_context")
    tasks = _all_tasks(env, benchmark)
    assert len(tasks) >= sum(protocol.SPLIT_SIZES[benchmark].values())
    searched = 0
    for task in tasks:
        needles = audit.CellAuditor._labels(env, task)
        needles |= {audit.normalise(a) for a in audit._answers(task)}
        needles |= audit.CellAuditor._taxonomy(env, task)
        for needle in needles:
            searched += 1
            assert next(corpus.find(needle), None) is None, (task.raw_id, needle)
    assert searched >= len(tasks)


def test_an_agent_context_with_a_label_is_found_by_the_audits_search():
    """The search above can fail: a raw id put into the text is found."""
    env = alf.AlfWorldEnv(backend_factory=Games().factory())
    task = env.load_tasks("train")[0]
    corpus = audit.Corpus()
    corpus.add(
        audit.normalise(env.agent_context() + f"\n(for {task.raw_id})"), "x", "x"
    )
    assert any(
        next(corpus.find(n), None) is not None
        for n in audit.CellAuditor._labels(env, task)
    )


# --- the helper module ----------------------------------------------------------------------


def test_fill_fields_keeps_unnamed_placeholders_and_refuses_what_would_not_survive():
    assert ac.fill_fields("a {x} b {y} {x}", x=1) == "a 1 b {y} 1"
    with pytest.raises(ValueError, match="literal braces"):
        ac.fill_fields("a {{x}}")
    with pytest.raises(ValueError, match="plain name"):
        ac.fill_fields("{reward:.2f}")
    with pytest.raises(ValueError, match="plain name"):
        ac.fill_fields("{x!r}")
    assert ac.fields_of("{a} {b} {a}") == ("a", "b")


def test_single_turn_sections_follow_the_history_length():
    titles = [
        s.title for s in ac.single_turn_sections("user", "F", "L {history_length}", 3)
    ]
    assert titles == [
        "user message, step 1",
        "user message, step 2",
        "user message, step 3",
        "user message, step 4 and later",
    ]
    assert [
        s.template
        for s in ac.single_turn_sections("user", "F", "L {history_length}", 3)
    ] == ["F", "L 1", "L 2", "L 3"]
    (only,) = ac.single_turn_sections("user", "F", "L", 0)
    assert only.title == "user message, every step" and only.sent_at(99)
    assert ac.PromptSection("user", 2, 5, "x").title == "user message, steps 2 to 5"


def test_render_refuses_a_slot_without_a_description_and_a_description_without_a_slot():
    section = ac.PromptSection("user", 1, None, "{a} and {b}")
    with pytest.raises(ValueError, match=r"without a description: \['b'\]"):
        ac.render([section], {"a": "x"})
    with pytest.raises(ValueError, match=r"not in the prompt: \['c'\]"):
        ac.render([section], {"a": "x", "b": "y", "c": "z"})
    text = ac.render([section], {"a": "x", "b": "y"}, notes=("A note.",))
    assert text == (
        f"{ac.INTRO}\n- {{a}}: x\n- {{b}}: y\nA note.\n\n"
        "=== user message, every step ===\n{a} and {b}"
    )


# --- the rollout hands it to methods, in both phases ------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["learning", "test"])
async def test_the_rollout_hands_webshops_agent_context_to_the_method(tmp_path, phase):
    shop = make_shop(tmp_path / "shop")
    env = shop.env()
    task = env.load_tasks("train")[0]

    def shopper(endpoint: str, body: dict[str, Any]) -> Any:
        return text_reply(respond(shopper_action(body["messages"], right=True)))

    method = RecordingMethod("")
    fake = FakeUpstream(responder=shopper)
    async with gateway_actor(tmp_path, fake) as (_gateway, actor):
        run = await ro.run_episode(
            env,
            task,
            method=method,
            actor=actor,
            salt=SALT,
            phase=phase,
            idle_timeout_s=5.0 if phase == "learning" else None,
        )
    view = method.events[0][1]
    assert isinstance(view, TaskView)
    assert view.agent_context == env.agent_context() == ws.agent_context(env.templates)
    assert run.trajectory.agent_context == view.agent_context
    assert run.trajectory.to_dict()["agent_context"] == view.agent_context
    # The messages methods receive are unchanged by it (Q20).
    assert "{current_observation}" not in str(run.trajectory.messages)


def test_the_context_record_is_the_texts_digest():
    """The framing is harness code outside the protocol fingerprint,
    so each attempt records the text's digest."""
    assert ac.context_record("") == {
        "sha256": hashlib.sha256(b"").hexdigest(),
        "chars": 0,
    }
    text = ws.agent_context(ws.WebShopTemplates.load())
    assert ac.context_record(text) == {
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "chars": len(text),
    }


def test_every_attempt_records_its_agent_context(tmp_path):
    """run.py records the attempt's agent context in ``attempt.json`` (its digest and
    length), and the report reads it back (``report.cell_agent_context``)."""
    from evals.skill_bench import report
    from evals.skill_bench.cell_records import CellPaths, read_json
    from evals.skill_bench.tests.e2e_support import (
        E2EModel,
        alfworld_env,
        cell_spec,
        run_cells,
    )
    from evals.skill_bench.tests.fakes import serve_gateway

    env = alfworld_env(tmp_path)
    spec = cell_spec("alfworld", "none", n_train=1, n_test=1)
    runs = tmp_path / "runs"
    fake = FakeUpstream(responder=E2EModel())
    with serve_gateway(
        tmp_path, fake, backoff_base_s=protocol.GATEWAY_BACKOFF_BASE_S
    ) as (_gw, url):
        outcome = asyncio.run(run_cells(runs, url, {"alfworld": env}, [spec]))
    assert outcome[spec.cell_id].status == "complete"
    expected = ac.context_record(ac.env_agent_context(env))
    assert expected["chars"] > 0
    state = read_json(CellPaths(runs / spec.cell_id).attempt(1).state)
    assert state["agent_context"] == expected
    cell = report.load_cell(runs / spec.cell_id, runs)
    assert report.cell_agent_context(cell) == expected
    # An attempt from before the record has none.
    cell.attempts[-1].state.pop("agent_context")
    assert report.cell_agent_context(cell) is None


def test_an_env_without_an_agent_context_gives_an_empty_one():
    class Bare:
        pass

    class Wrong:
        def agent_context(self) -> Any:
            return 3

    assert ac.env_agent_context(Bare()) == ""
    with pytest.raises(TypeError, match="returned int"):
        ac.env_agent_context(Wrong())
