"""WebShop environment: prompts, guards, scoring, the client.

Offline: episodes run against the image's real ``server.py`` service over the scripted
shop of ``tests/webshop_support.py`` (an ``httpx`` transport into ``server.route``), so
the guards are checked on both sides of the wire. Server-only behaviour (the lock,
sessions, the manifest check, the HTTP layer, prepare mode) is in
``test_webshop_server.py``.

Set ``VERL_AGENT_DIR`` (langfengQ/verl-agent at 20bd331) and ``WEBSHOP_UPSTREAM_DIR``
(princeton-nlp/WebShop at 64fa2a5) to compare the frozen excerpts with the clones; those
tests skip without them.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import httpx
import pytest

from evals.skill_bench import protocol
from evals.skill_bench.envs import agent_context as ac
from evals.skill_bench.envs import webshop as ws
from evals.skill_bench.envs.base import Task
from evals.skill_bench.envs.vendor.webshop_projection import webshop_projection
from evals.skill_bench.infra import failures
from evals.skill_bench.infra.failures import FailureKind, classify_exception
from evals.skill_bench.tests.webshop_support import (
    DONE_PAGE,
    INDEX_SHA256,
    SHOP_URL,
    Shop,
    make_shop,
    respond,
    server_module,
    shopper_action,
    synthetic_goal,
)

ROOT = protocol.SKILL_BENCH_DIR
VERL_AGENT_DIR = os.environ.get("VERL_AGENT_DIR")
WEBSHOP_UPSTREAM_DIR = os.environ.get("WEBSHOP_UPSTREAM_DIR")
requires_verl_agent = pytest.mark.skipif(
    not VERL_AGENT_DIR, reason="VERL_AGENT_DIR (verl-agent 20bd331 clone) not set"
)
requires_upstream = pytest.mark.skipif(
    not WEBSHOP_UPSTREAM_DIR, reason="WEBSHOP_UPSTREAM_DIR (WebShop clone) not set"
)


@pytest.fixture(scope="module")
def t() -> ws.WebShopTemplates:
    return ws.WebShopTemplates.load()


@pytest.fixture
def shop(tmp_path: Path) -> Shop:
    return make_shop(tmp_path)


def no_network(base_url: str, timeout_s: float) -> httpx.AsyncClient:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"the audit host made a request: {request.url}")

    return httpx.AsyncClient(base_url=base_url, transport=httpx.MockTransport(refuse))


def task_of(env: ws.WebShopEnv, split: str = "test", index: int = 0) -> Task:
    return env.load_tasks(split)[index]  # type: ignore[arg-type]


async def play(
    env: ws.WebShopEnv, task: Task, *, right: bool = True, limit: int = 40
) -> tuple[Any, list[Any]]:
    """Reset and play the scripted shopper to the end; ``(reset, steps)``."""
    episode = env.episode(task)
    async with episode:
        reset = await episode.reset()
        fields, steps = reset.fields, []
        for _ in range(limit):
            request = env.actor_messages(fields)
            step = await episode.step(respond(shopper_action(request, right=right)))
            steps.append(step)
            fields = step.fields
            if step.done:
                break
    return reset, steps


# ---------------------------------------------------------------------------
# Frozen templates and their provenance
# ---------------------------------------------------------------------------


def _format_spec() -> dict[str, Any]:
    return json.loads((ROOT / ws.TEMPLATE_DIR / "format.json").read_text())


def _literals(excerpt: str) -> dict[str, str]:
    module = ast.parse(excerpt)
    return {
        node.targets[0].id: node.value.value  # type: ignore[attr-defined]
        for node in module.body
        if isinstance(node, ast.Assign)
    }


def test_the_templates_are_verl_agents_literals_byte_for_byte(t):
    """The .md files are WEBSHOP_TEMPLATE_NO_HIS / WEBSHOP_TEMPLATE of verl-agent's
    ``prompts/webshop.py`` (the excerpt format.json quotes), U+2011 and trailing spaces
    included."""
    literals = _literals(_format_spec()["provenance"]["templates_source"]["excerpt"])
    assert t.no_history == literals["WEBSHOP_TEMPLATE_NO_HIS"]
    assert t.with_history == literals["WEBSHOP_TEMPLATE"]
    assert "e‑commerce" in t.no_history and "are: \n[" in t.no_history
    # The prompt never states the step cap.
    for text in (t.no_history, t.with_history, t.turn):
        assert str(protocol.WEBSHOP_MAX_STEPS) not in text
        assert "step limit" not in text.lower()


def test_the_turn_template_is_verl_agents_with_three_lines_removed(t):
    """``rollout_turn.md`` is WEBSHOP_TEMPLATE without its leading
    blank line, its persona line and its "Prior to this step" line; every other line is
    byte-exact, trailing spaces included (``format.json`` quotes the literal)."""
    source = _format_spec()["provenance"]["turn_file"]
    literal = _literals(source["excerpt"])["WEBSHOP_TEMPLATE"]
    assert literal == t.with_history
    lines = literal.split("\n")
    assert lines[0] == ""
    assert lines[1].startswith("You are an expert autonomous agent")
    assert lines[3].startswith("Prior to this step, you have already taken")
    assert t.turn == "\n".join([lines[2], *lines[4:]])
    assert t.turn.startswith(
        "Your task is to: {task_description}.\nYou are now at step"
    )
    assert "are: \n[" in t.turn and "tags. \nOnce" in t.turn  # trailing spaces kept
    assert ac.fields_of(t.turn) == (
        "task_description",
        "current_step",
        "current_observation",
        "available_actions",
    )
    assert "e‑commerce" not in t.turn and "step(s)" not in t.turn
    assert (ROOT / ws.TEMPLATE_DIR / "rollout_turn.md").read_text() == t.turn


def test_the_format_values(t):
    # The whole episode, one conversation: no window, no 13,000-character fallback.
    assert t.history == ws.FULL_CONVERSATION == "full_conversation"
    assert t.no_reply_turn == "(no reply)"
    templates = _format_spec()["templates"]
    for gone in (
        "history_length",
        "history_entry",
        "history_joiner",
        "history_fallback_chars",
    ):
        assert gone not in templates and gone not in _format_spec()["provenance"]
    assert (
        "append-only conversation"
        in (_format_spec()["provenance"]["history"]["change"])
    )
    assert (
        "skill_bench's own text"
        in (_format_spec()["provenance"]["no_reply_turn"]["change"])
    )
    assert t.available_excluded_clickables == ("search",)
    assert t.special_search_pattern == "<[a-z]>"
    assert t.final_message == (
        "Thank you for shopping with us! Your score (min 0.0, max 1.0): {reward:.2f}"
    )
    assert t.cap_message == (
        "The episode ended at the step limit without a purchase. "
        "Your score (min 0.0, max 1.0): {reward:.2f}"
    )
    assert str(protocol.WEBSHOP_MAX_STEPS) not in t.cap_message
    assert t.noop_action == ""
    spec = _format_spec()
    assert spec["verl_agent"]["commit"] == "20bd331bdbc9026a5668e11362178e10ab7400c8"
    assert spec["webshop"]["commit"] == protocol.WEBSHOP_SERVER["upstream_commit"]


def test_the_step_cap_is_verl_agents_webshop_default(t):
    """15 steps, as verl-agent 20bd331 runs WebShop;
    ``format.json`` quotes where it comes from (an example script's
    ``env.max_steps=15``)."""
    assert protocol.WEBSHOP_MAX_STEPS == 15
    cap = _format_spec()["provenance"]["max_steps"]
    assert cap["repo"] == "verl_agent" and cap["file"].endswith("run_webshop.sh")
    assert f"env.max_steps={protocol.WEBSHOP_MAX_STEPS} " in cap["excerpt"]
    assert "verl-agent" in cap["change"]


@requires_verl_agent
def test_every_verl_agent_webshop_script_runs_15_steps():
    scripts = sorted(Path(str(VERL_AGENT_DIR)).glob("examples/*/run_webshop*.sh"))
    assert len(scripts) >= 5
    for script in scripts:
        text = script.read_text()
        assert f"env.max_steps={protocol.WEBSHOP_MAX_STEPS} " in text, script


@pytest.mark.parametrize(
    ("repo", "env_dir"),
    [
        pytest.param("verl_agent", VERL_AGENT_DIR, marks=requires_verl_agent),
        pytest.param("webshop", WEBSHOP_UPSTREAM_DIR, marks=requires_upstream),
    ],
)
def test_format_json_excerpts_are_verbatim_source_lines(repo, env_dir):
    spec = _format_spec()
    seen = 0
    for key, source in spec["provenance"].items():
        if source.get("repo") != repo or source.get("file") is None:
            continue
        data = (Path(env_dir) / source["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == source["file_sha256"], key
        first, last = source["lines"]
        lines = data.decode("utf-8").splitlines(keepends=True)
        assert "".join(lines[first - 1 : last]) == source["excerpt"], key
        seen += 1
    assert seen


@requires_verl_agent
def test_the_vendored_projection_is_the_original_logic():
    import importlib.util

    path = (
        Path(str(VERL_AGENT_DIR))
        / "agent_system/environments/env_package/webshop/projection.py"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "0808d4643459acf2a4c496eab3fc8735025c440f28cc517066355a1f5fbc048f"
    )
    spec = importlib.util.spec_from_file_location("verl_projection", path)
    assert spec is not None and spec.loader is not None
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    assert original.webshop_projection(list(PROJECTION_CORPUS)) == webshop_projection(
        list(PROJECTION_CORPUS)
    )


#: Responses and what verl-agent 20bd331's projection makes of them (computed with the
#: original).
PROJECTION_CORPUS = (
    "<think>ok</think><action>search[red shoes]</action>",
    "<think>ok</think><action> Click[Buy Now] </action>",
    "<action>click[b0123]</action>",
    "<think>x</think><action></action>",
    "<think>x</think>no tags at all",
    "<think>中文</think><action>click[buy now]</action>",
    "<THINK>x</THINK><ACTION>search[A B]</ACTION>",
    "<think>a</think><action>search[a]</action><action>search[b]</action>",
    "</action><action>click[x]",
    "<think>a<action>click[< prev]</action></think>",
    "",
)
PROJECTION_EXPECTED = (
    [
        "search[red shoes]",
        "click[buy now]",
        "click[b0123]",
        "",
        "think>no tags at all",
        "click[buy now]",
        "search[a b]",
        "search[a]",
        "",
        "click[< prev]",
        "",
    ],
    [1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
)


def test_the_vendored_projection_on_the_corpus():
    assert webshop_projection(list(PROJECTION_CORPUS)) == PROJECTION_EXPECTED


def test_no_model_visible_literal_in_the_module():
    """Model-visible text lives in the fingerprinted templates (like the other envs): no
    string constant of the module's code (docstrings aside) carries any of it."""
    tree = ast.parse((ROOT / "envs" / "webshop.py").read_text())
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    constants = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]
    for literal in (
        "Thank you for shopping",
        "Your score",
        "Available actions",
        "search[<your query>]",
        "<action>",
        "Instruction:",
        " [SEP] ",
        "click[",
    ):
        assert not any(literal in c for c in constants), literal


# ---------------------------------------------------------------------------
# Rendering (verl-agent build_text_obs)
# ---------------------------------------------------------------------------

PAGE = "WebShop [SEP] Instruction: [SEP] i need red shoes, and price lower than 40.00 dollars [SEP] Search"
TASK = "i need red shoes, and price lower than 40.00 dollars"


def test_available_actions_exclude_click_search_and_unknown_keys(t):
    actions = ws.list_available_actions(
        t, {"has_search_bar": True, "clickables": ["search", "back to search"]}
    )
    assert actions == ("search[<your query>]", "click[back to search]")
    assert ws.format_available_actions(t, actions) == (
        "'search[<your query>]',\n'click[back to search]',"
    )
    with pytest.raises(ws.WebShopEnvError, match="unknown keys"):
        ws.list_available_actions(t, {"has_search_bar": True, "clickables": [], "x": 1})
    with pytest.raises(ws.WebShopEnvError, match="clickables"):
        ws.list_available_actions(t, {"has_search_bar": True, "clickables": "search"})


def test_task_and_observation_formatting(t):
    assert ws.extract_task(t, PAGE) == TASK
    with pytest.raises(ws.WebShopEnvError):
        ws.extract_task(t, "Back to Search [SEP] Page 1")
    later = f"WebShop [SEP] Instruction: [SEP] {TASK} [SEP] Back to Search [SEP] B01"
    assert ws.format_observation(t, later, TASK) == "'Back to Search' [SEP] 'B01'"
    # A page without the instruction stays as it is (verl-agent format_obs).
    assert (
        ws.format_observation(t, "no instruction here", TASK) == "no instruction here"
    )


def _fields(*conversation: str, steps: int | None = None) -> dict[str, Any]:
    """Fields holding ``conversation``: user and assistant texts in turn."""
    messages = tuple(
        {"role": "assistant" if i % 2 else "user", "content": text}
        for i, text in enumerate(conversation)
    )
    return {
        "observation": "'Search'",
        "available_actions": ("search[<your query>]",),
        "task_description": TASK,
        "conversation": messages,
        "step_count": (len(messages) - 1) // 2 if steps is None else steps,
    }


def test_the_first_prompt_is_the_no_history_template(t):
    prompt = ws.first_prompt(t, TASK, "'Search'", ("search[<your query>]",))
    assert prompt == t.no_history.format(
        task_description=TASK,
        current_observation="'Search'",
        available_actions="'search[<your query>]',",
    )
    messages = ws.render_actor_messages(t, _fields(prompt), "remember X")
    assert messages == [{"role": "user", "content": "remember X\n\n" + prompt}]
    # An empty or whitespace memory injects nothing (Q21).
    assert ws.render_actor_messages(t, _fields(prompt), "  \n") == [
        {"role": "user", "content": prompt}
    ]


def test_a_later_step_appends_the_reply_and_the_turn_template(t):
    """Step k's request is step k-1's, then that step's reply and
    ``rollout_turn.md`` with step k's number, page and actions; the memory stays in the
    first message."""
    first = ws.first_prompt(t, TASK, "'Search'", ("search[<your query>]",))
    page = "'Back to Search' [SEP] 'B01'"
    turn = ws.turn_prompt(t, TASK, 2, page, ("click[back to search]", "click[b01]"))
    assert turn == t.turn.format(
        task_description=TASK,
        current_step=2,
        current_observation=page,
        available_actions="'click[back to search]',\n'click[b01]',",
    )
    assert turn.startswith(f"Your task is to: {TASK}.\nYou are now at step 2 and ")
    reply = "<think>a</think><action>search[red shoes]</action>"
    messages = ws.render_actor_messages(t, _fields(first, reply, turn), "remember X")
    assert messages == [
        {"role": "user", "content": "remember X\n\n" + first},
        {"role": "assistant", "content": reply},
        {"role": "user", "content": turn},
    ]
    assert ws.render_actor_messages(t, _fields(first, reply, turn))[1:] == messages[1:]


def test_replies_are_kept_verbatim_or_become_the_placeholder(t):
    """The assistant message is the reply as the episode received it (lone surrogates
    replaced); a call without a reply, or an empty or whitespace-only reply, is
    ``no_reply_turn``, skill_bench's own text."""
    kept = "  <think>x</think><action>click[a]</action>\n"
    assert ws.reply_turn(t, kept) == kept
    assert ws.reply_turn(t, "no tags at all") == "no tags at all"
    assert ws.reply_turn(t, "<action>search[\ud83d x]</action>") == (
        "<action>search[\ufffd x]</action>"
    )
    for nothing in ("", "  \n\t", {ws.NO_REPLY_KEY: True}):
        assert ws.reply_turn(t, nothing) == t.no_reply_turn == "(no reply)"


def test_fields_without_their_whole_conversation_are_refused(t):
    """An episode whose fields lost the conversation, or part of it, fails loudly: the
    actor is never sent a shorter episode. Nor is a call made after the last step."""
    first = ws.first_prompt(t, TASK, "'Search'", ("search[<your query>]",))
    assert len(ws.render_actor_messages(t, _fields(first, "r1", "turn 2"))) == 3
    lost = {k: v for k, v in _fields(first).items() if k != "conversation"}
    with pytest.raises(ws.WebShopEnvError, match="no conversation"):
        ws.render_actor_messages(t, lost)
    with pytest.raises(ws.WebShopEnvError, match="part of the episode is lost"):
        ws.render_actor_messages(t, _fields(first, steps=2))
    with pytest.raises(ws.WebShopEnvError, match="is over"):
        ws.render_actor_messages(t, _fields(first, "r1", steps=1))
    two_users = {
        **_fields(first, "r1", "turn 2"),
        "conversation": tuple({"role": "user", "content": x} for x in "abc"),
    }
    with pytest.raises(ws.WebShopEnvError, match="message 1 should be the assistant"):
        ws.render_actor_messages(t, two_users)
    with pytest.raises(ws.WebShopEnvError, match="step_count"):
        ws.render_actor_messages(t, {**_fields(first), "step_count": None})


# ---------------------------------------------------------------------------
# Responses -> actions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "command", "format_error", "valid"),
    [
        (
            "<think>a</think><action>search[red shoes]</action>",
            "search[red shoes]",
            False,
            True,
        ),
        ("<action>Click[Buy Now]</action>", "click[buy now]", False, False),
        ("<think>中</think><action>click[x]</action>", "click[x]", False, False),
        ("I would buy it", None, True, False),
        ("<think>a</think><action>   </action>", None, True, False),
        ("<think>a</think><action></action>", None, True, False),
        ("", None, True, False),
        (None, None, True, False),
        # The tag check reads the tags in any case, as the projection does.
        ("<ACTION>search[x]</ACTION>", "search[x]", False, False),
        (
            "<think>a</think><Action>click[buy now]</Action><action></action>",
            "click[buy now]",
            False,
            True,
        ),
        ("<think>a</think></action><action>click[buy now]", None, True, False),
        (
            "<think>a</think><action></action><action>click[x]</action>",
            None,
            True,
            False,
        ),
    ],
)
def test_parse_response(t, raw, command, format_error, valid):
    assert ws.parse_response(t, raw) == ws.ParsedResponse(command, format_error, valid)


def test_the_tag_check_and_the_projection_always_agree(t):
    """Whenever the check finds a pair the projection extracts from it (its
    fallback, the reply's last 20 characters, never runs), and a reply the check refuses
    is one the projection finds no pair in, or an empty one."""
    for raw in (*PROJECTION_CORPUS, "<Action>x</ACTION>", "<aCtIoN>y</action>"):
        response = raw.strip()
        tagged = re.search(t.action_tag_pattern, response, re.DOTALL) is not None
        lowered = response.lower()
        start, end = lowered.find("<action>"), lowered.find("</action>")
        assert tagged == (start != -1 and end != -1 and start < end), raw
        if tagged:
            parsed = ws.parse_response(t, raw)
            command = webshop_projection([response])[0][0].strip()
            assert parsed.command == (command or None), raw


def test_an_action_that_is_not_utf8_is_never_sent(t):
    """A lone surrogate (a broken escape in a provider's JSON) cannot be put
    in a JSON request; the action is refused, and kept for the history with U+FFFD."""
    parsed = ws.parse_response(
        t, "<think>a</think><action>search[\ud83d shoes]</action>"
    )
    assert parsed == ws.ParsedResponse(
        "search[\ufffd shoes]", False, True, ws.REJECTED_UNENCODABLE
    )
    parsed.command.encode("utf-8")  # the history's copy is valid UTF-8
    assert ws.without_surrogates("a\udc80b\U0001f600") == "a\ufffdb\U0001f600"


@pytest.mark.parametrize(
    ("command", "refused"),
    [
        ("search[<r> anything]", True),
        ("search[<a>]", True),
        ("search[ <q> red shoes]", True),
        ("search[<C> red]", True),  # upstream lower-cases the query
        ("search[red <r> shoes]", False),
        ("search[<rr> shoes]", False),
        ("click[<r>]", False),
        ("search[red shoes]", False),
        ("search[]", False),
    ],
)
def test_special_searches_are_refused_on_both_sides(t, command, refused):
    assert ws.refused_search(t, command) is refused
    assert server_module().refused_search(command) is refused


# ---------------------------------------------------------------------------
# Episodes against the scripted shop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_purchase_ends_with_the_score_line_only(shop):
    env = shop.env()
    task = task_of(env, "test", 1)
    reset, steps = await play(env, task)
    first = reset.messages[0]["content"]
    assert reset.instruction in first and first.startswith("WebShop [SEP] Instruction:")
    assert first.endswith("Available actions:\nsearch[<your query>]")
    last = steps[-1]
    # The scripted shopper's step count depends on the item's options: search, open,
    # one click per option, Buy Now.
    assert last.done and last.private == {
        "reward": 1.0,
        "purchased": True,
        "steps": len(steps),
    }
    assert len(steps) == 3 + len(task.raw["goal_options"])
    assert last.messages == (
        {
            "role": "user",
            "content": "Thank you for shopping with us! Your score (min 0.0, max 1.0): 1.00",
        },
    )
    score = env.score(task, None, last.private)
    assert score.primary == 1.0 and score.answered
    expected = {"success": 1.0, "success_within_15": 1.0}
    if env.manifest_status() == ws.MANIFEST_FINAL:
        # A final manifest gives every item its best possible reward.
        expected["best_reward"] = float(task.raw["best_reward"])
    assert dict(score.metrics) == expected
    # Every later message is the page without the instruction, then the actions.
    for step in steps[:-1]:
        content = step.messages[0]["content"]
        assert (
            reset.instruction not in content and "\n\nAvailable actions:\n" in content
        )


@pytest.mark.asyncio
async def test_the_final_message_holds_no_target_field(shop):
    """Neither the done page nor any goal field reaches a method; the server never
    returns the done page (its hidden blocks keep the goal)."""
    env = shop.env()
    task = task_of(env, "test", 1)
    _, steps = await play(env, task, right=False)
    final = steps[-1].messages[0]["content"]
    assert final.endswith(": 0.00")
    goal = shop.backend.goals[int(task.raw["goal_index"])]
    for value in (  # nothing of the goal in the score line
        goal["asin"],
        goal["category"],
        goal["query"],
        goal["product_category"],
        str(goal["price_upper"]),
        *goal["attributes"],
        *goal["goal_options"],
    ):
        assert str(value).lower() not in final.lower(), value
    # What the server sent back after Buy Now: the reward, no page, no actions.
    last_reply = [r for r in shop.replies if r[1].endswith("/step")][-1][3]
    assert last_reply == {
        "observation": None,
        "reward": 0.0,
        "done": True,
        "accepted": True,
        "available_actions": None,
        "shop_error": None,
    }
    for _method, _path, _status, payload in shop.replies:
        assert "Reward Details" not in json.dumps(payload)
    assert env.leak_problems([*_messages(steps)]) == []


def _messages(steps: list[Any]) -> list[dict[str, Any]]:
    return [m for s in steps for m in s.messages]


def test_the_structural_leak_check(t):
    final = ws.final_message(t, 0.5)
    page = {"role": "user", "content": "'Back to Search' [SEP] 'B01'"}
    assert ws.leak_problems(t, [page, final]) == []
    assert ws.leak_problems(t, [final, page]) == [
        (0, "the score line is not the last message")
    ]
    done = DONE_PAGE.format(
        asin="B0",
        options=[],
        attributes=[],
        category="c",
        query="q",
        product_category="p",
        price_upper=1,
        instruction_text="i",
        reward=1.0,
    )
    problems = ws.leak_problems(t, [{"role": "user", "content": done}])
    assert problems and problems[0][0] == 0
    sep = t.observation_separator
    for marker in t.leak_markers:
        joined = sep.join(marker)
        # Raw or formatted (format_obs quotes each part), first, middle or last part.
        for page in (
            f"x {sep}{joined}{sep}y",
            sep.join(f"'{p}'" for p in ("x", *marker, "y")),
            f"{joined}{sep}y",
            ws.observation_message(
                t, f"'x'{sep}" + sep.join(f"'{p}'" for p in marker), ["click[x]"]
            )["content"],
        ):
            assert ws.leak_problems(t, [{"role": "user", "content": page}]), page
    # Each marker alone is enough: the done page's hidden blocks, the score block.
    for part in ("instuction text", "price upper", "Reward Details"):
        assert ws.leak_problems(t, [{"role": "user", "content": f"'a'{sep}'{part}'"}])
    # The model's own words are not the environment's.
    assert (
        ws.leak_problems(t, [{"role": "assistant", "content": "Reward Details?"}]) == []
    )
    assert ws.leak_problems(t, [{"role": "tool", "content": final["content"]}]) == [
        (0, "the score line is not a user message")
    ]
    # A score line with more text is not the score line.
    assert ws.leak_problems(
        t, [{"role": "user", "content": final["content"] + " asin B0"}]
    ) == [(0, "it holds the score line among other text")]
    assert ws.final_message_pattern(t).fullmatch(ws.final_message(t, 1.0)["content"])
    assert not ws.final_message_pattern(t).fullmatch(
        "Thank you for shopping with us! Your score (min 0.0, max 1.0): 1.5"
    )
    # The step cap's line closes an episode like the score line.
    cap = ws.cap_message(t)
    last_page = {"role": "user", "content": "'Back to Search' [SEP] 'B01'"}
    assert cap["content"].endswith("Your score (min 0.0, max 1.0): 0.00")
    assert ws.leak_problems(t, [last_page, cap]) == []
    assert ws.leak_problems(t, [cap, last_page]) == [
        (0, "the score line is not the last message")
    ]
    assert ws.leak_problems(t, [{"role": "tool", "content": cap["content"]}]) == [
        (0, "the score line is not a user message")
    ]
    assert ws.leak_problems(
        t, [{"role": "user", "content": f"'a'{sep}" + cap["content"]}]
    ) == [(0, "it holds the score line among other text")]


@pytest.mark.asyncio
async def test_a_missing_or_empty_tag_is_a_frozen_noop(shop, t):
    """Nothing reaches the server, the page stays, the step counts, and the
    conversation keeps each reply (an empty one as the placeholder)."""
    env = shop.env()
    episode = env.episode(task_of(env))
    async with episode:
        reset = await episode.reset()
        before = len(shop.backend.actions)
        raws = ("I am not sure.", "<think>x</think><action> </action>", "")
        for raw in raws:
            step = await episode.step(raw)
            assert step.fields["format_error"] is True and step.fields["action"] == ""
            assert step.fields["accepted"] is False and not step.done
            assert step.messages[0]["content"].startswith("'Search'")
        assert len(shop.backend.actions) == before
        assert episode.steps == 3 and episode.format_errors == 3
        request = env.actor_messages(step.fields)
        assert [m["content"] for m in request[1::2]] == [*raws[:2], t.no_reply_turn]
        assert all(reset.instruction in m["content"] for m in request[::2])
        assert "You are now at step 4 and" in request[-1]["content"]


@pytest.mark.asyncio
async def test_a_call_without_a_reply_is_a_noop_without_a_format_error(shop):
    env = shop.env()
    episode = env.episode(task_of(env))
    async with episode:
        await episode.reset()
        step = await episode.step(env.call_error_response())
        assert step.fields["format_error"] is False and step.fields["action"] == ""
        assert episode.steps == 1 and episode.format_errors == 0
        # The conversation shows the placeholder for it.
        reply = env.actor_messages(step.fields)[1]
        assert reply == {"role": "assistant", "content": env.templates.no_reply_turn}
        with pytest.raises(TypeError):
            await episode.step({"other": True})  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_special_search_never_reaches_the_shop(shop):
    env = shop.env()
    episode = env.episode(task_of(env))
    async with episode:
        await episode.reset()
        step = await episode.step(respond("search[<r> red shoes]"))
        assert step.fields["action"] == "search[<r> red shoes]"
        assert step.fields["accepted"] is False and step.fields["format_error"] is False
        assert step.messages[0]["content"].startswith("'Search'")
    assert not any(path.endswith("/step") for _, path, _ in shop.requests)
    assert shop.backend.special_searches == []


@pytest.mark.asyncio
async def test_why_an_action_was_not_processed_is_recorded(shop):
    """Each action that leaves the page without the shop's own refusal says why
    (``rejected``, a recorded field): a special search and an action that is not valid
    UTF-8 are never sent; one the shop raised on was sent, and the server restored the
    session (``shop_error``)."""
    env = shop.env()
    episode = env.episode(task_of(env))
    async with episode:
        await episode.reset()
        step = await episode.step(respond("search[<r> red shoes]"))
        assert step.fields["rejected"] == ws.REJECTED_SPECIAL_SEARCH
        step = await episode.step(respond("search[\ud83d shoes]"))
        assert step.fields["rejected"] == ws.REJECTED_UNENCODABLE
        assert step.fields["action"] == "search[\ufffd shoes]"
        assert step.fields["accepted"] is False and step.fields["shop_error"] is None
        assert not any(path.endswith("/step") for _, path, _ in shop.requests)
        on_search = await episode.step(respond("search[gold]"))
        assert on_search.fields["rejected"] is None
        assert on_search.fields["shop_error"] is None
        shop.backend.fail_next_step = KeyError("B0LOWER")
        step = await episode.step(respond("click[b0lower]"))
        assert step.fields["rejected"] == ws.REJECTED_SHOP_ERROR
        assert step.fields["shop_error"] == "KeyError: 'B0LOWER'"
        assert step.fields["accepted"] is False and not step.done
        assert step.messages == on_search.messages  # the page stays
        request = env.actor_messages(step.fields)
        # The conversation keeps every reply, the unencodable one with U+FFFD.
        assert request[3]["content"] == respond("search[\ufffd shoes]")
        assert request[7]["content"] == respond("click[b0lower]")
        step = await episode.step(respond("click[back to search]"))
        assert step.fields["accepted"] is True and step.fields["rejected"] is None
    driver_fields = ws_driver_fields()
    assert {"rejected", "shop_error"} <= set(driver_fields)


def ws_driver_fields() -> tuple[str, ...]:
    from evals.skill_bench.agent import rollout

    return rollout._DRIVERS["webshop"].record_fields


@pytest.mark.asyncio
async def test_a_shop_error_on_an_accepted_action_breaks_the_episode(shop):
    """A reply that names a shop error must leave the page: anything else is a server
    this harness does not understand (HARNESS), never a silent step."""
    inner = shop.client_factory()

    def lying(base_url: str, timeout_s: float) -> httpx.AsyncClient:
        client = inner(base_url, timeout_s)
        send = client.send

        async def patched(request: httpx.Request, **kw: Any) -> httpx.Response:
            response = await send(request, **kw)
            if not request.url.path.endswith("/step"):
                return response
            body = json.loads(response.content)
            body.update(shop_error="KeyError: 'x'", accepted=True)
            return httpx.Response(200, json=body)

        client.send = patched  # type: ignore[method-assign]
        return client

    env = ws.WebShopEnv(server_url=SHOP_URL, client_factory=lying)
    episode = env.episode(task_of(env))
    async with episode:
        await episode.reset()
        with pytest.raises(ws.WebShopEnvError, match="shop error"):
            await episode.step(respond("search[gold]"))
        with pytest.raises(ws.WebShopServerError, match="interrupted or failed"):
            await episode.step(respond("search[gold]"))


@pytest.mark.asyncio
async def test_an_action_the_shop_does_not_accept_leaves_the_page(shop):
    env = shop.env()
    episode = env.episode(task_of(env))
    async with episode:
        await episode.reset()
        step = await episode.step(respond("click[nothing here]"))
        assert step.fields["accepted"] is False
        assert step.messages[0]["content"].startswith("'Search'")
        step = await episode.step(respond("click[search]"))  # upstream refuses it
        assert step.fields["accepted"] is False
        step = await episode.step(respond("search[red shoes]"))
        assert step.fields["accepted"] is True
        assert "'Back to Search'" in step.messages[0]["content"]


@pytest.mark.asyncio
async def test_the_step_cap_ends_unanswered_without_a_score(shop):
    env = shop.env(max_steps=3)
    task = task_of(env)
    episode = env.episode(task)
    async with episode:
        await episode.reset()
        for n in range(3):
            step = await episode.step(respond("click[nothing]"))
        assert step.done and step.fields["truncated"] is True
        assert step.private == {"reward": None, "purchased": False, "steps": 3}
        assert all("Thank you" not in m["content"] for m in step.messages)
        # The last page, then the cap's 0.00 line, last.
        assert len(step.messages) == 2
        assert step.messages[0]["content"].startswith("'Search'")
        assert step.messages[-1] == ws.cap_message(env.templates)
        assert env.leak_problems(list(step.messages)) == []
        with pytest.raises(RuntimeError, match="over"):
            await episode.step(respond("search[x]"))
    score = env.score(task, None, step.private)
    assert (score.primary, score.answered) == (0.0, False)
    assert score.metrics["success"] == 0.0
    with pytest.raises(ValueError):
        ws.WebShopEnv(
            max_steps=protocol.WEBSHOP_MAX_STEPS + 1, client_factory=no_network
        )


@pytest.mark.asyncio
async def test_the_default_episode_ends_after_fifteen_steps(shop):
    """Without a purchase the 15th step ends the episode,
    unanswered; the 16th is refused."""
    env = shop.env()
    assert env.max_steps == protocol.WEBSHOP_MAX_STEPS == 15
    task = task_of(env)
    episode = env.episode(task)
    async with episode:
        await episode.reset()
        for n in range(1, 16):
            step = await episode.step(respond("click[nothing]"))
            assert step.done is (n == 15), n
            assert (step.messages[-1] == ws.cap_message(env.templates)) is (n == 15)
        assert step.fields["truncated"] is True
        assert step.private == {"reward": None, "purchased": False, "steps": 15}
        with pytest.raises(RuntimeError, match="over"):
            await episode.step(respond("search[x]"))


# ---------------------------------------------------------------------------
# The actor's conversation
# ---------------------------------------------------------------------------

#: A memory with braces, quotes and lines: it goes in verbatim, in the first message.
MEMORY = "## Skills\n- check the {size} first & 'quote' \"x\"\n- then buy"
#: A product title that makes a page longer than the old 13,000-character fallback.
LONG_TITLE = "a very long product title with many words " * 400


def _capped_replies(distractor: str) -> list[Any]:
    """Fifteen steps without a purchase: a call without a reply, an empty reply, pages
    longer than 13,000 characters (the distractor's results row and item page), an
    option selected, a detail page, refused and invalid actions."""
    return [
        respond("search[gold]", "Search first."),
        {ws.NO_REPLY_KEY: True},
        "",
        respond(f"click[{distractor.lower()}]"),
        respond("click[blue]", "Pick the colour."),
        respond("click[description]"),
        respond("click[< prev]"),
        respond("search[<r> x]"),
        "  just thinking, no action \n",
        respond("click[back to search]"),
        respond("search[red]"),
        *[respond("click[nothing]")] * 4,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("memory", ["", MEMORY], ids=["no_memory", "memory"])
async def test_each_request_is_the_previous_one_plus_one_exchange(shop, t, memory):
    """Over a scripted 15-step episode, request k is an exact
    prefix of request k+1, which adds step k's reply (as the episode received it, or the
    placeholder) and step k+1's ``rollout_turn.md``; no message is ever re-rendered,
    truncated or replaced, however long the pages (no 13,000-character fallback), and
    the memory is in the first message only."""
    env = shop.env()
    task = task_of(env)
    distractor = f"BX{int(task.raw['goal_index']):08d}"
    shop.backend.products[distractor].title = LONG_TITLE
    replies = _capped_replies(distractor)
    assert len(replies) == protocol.WEBSHOP_MAX_STEPS
    async with env.episode(task) as episode:
        reset = await episode.reset()
        fields, previous = reset.fields, None
        pages = [fields["observation"]]
        for k, raw in enumerate(replies, start=1):
            request = env.actor_messages(fields, memory)
            assert len(request) == 2 * k - 1
            if previous is None:
                assert request == [
                    {
                        "role": "user",
                        "content": ws.insert_memory(
                            t,
                            ws.first_prompt(
                                t,
                                reset.instruction,
                                pages[0],
                                fields["available_actions"],
                            ),
                            memory,
                        ),
                    }
                ]
            else:
                said, raw_before = request[-2], replies[k - 2]
                assert request[: len(previous)] == previous  # byte-identical prefix
                assert said == {
                    "role": "assistant",
                    "content": ws.reply_turn(t, raw_before),
                }
                assert request[-1] == {
                    "role": "user",
                    "content": ws.turn_prompt(
                        t,
                        reset.instruction,
                        k,
                        fields["observation"],
                        fields["available_actions"],
                    ),
                }
            previous = request
            step = await episode.step(raw)
            fields = step.fields
            pages.append(fields["observation"])
        assert step.done and fields["truncated"] is True
    text = "\n".join(m["content"] for m in previous)
    assert LONG_TITLE.strip() in text and len(text) > 3 * 13_000
    assert all(page in text for page in pages[:-1])  # every page shown, in full
    assert [m["content"] for m in previous[1::2]][:3] == [
        replies[0],
        t.no_reply_turn,
        t.no_reply_turn,
    ]
    assert previous[17]["content"] == replies[8]  # kept verbatim, whitespace and all
    # The memory heads the first message only.
    if memory:
        assert previous[0]["content"].startswith(memory + "\n\n")
        assert all(memory not in m["content"] for m in previous[1:])
    # After the last step the conversation ends with its reply and no call follows.
    conversation = fields["conversation"]
    assert len(conversation) == 2 * protocol.WEBSHOP_MAX_STEPS
    assert [dict(m) for m in conversation[1:-1]] == [
        {"role": m["role"], "content": m["content"]} for m in previous[1:]
    ]
    assert dict(conversation[-1]) == {"role": "assistant", "content": replies[-1]}
    with pytest.raises(ws.WebShopEnvError, match="is over"):
        env.actor_messages(fields, memory)


def _never_the_score_or_cap_line(t: ws.WebShopTemplates, messages: Any) -> None:
    cap = ws.cap_message(t)["content"]
    for message in messages:
        content = message["content"]
        assert not ws.final_message_pattern(t).search(content), content
        assert cap not in content and "Your score" not in content, content
    users = [dict(m) for m in messages if m["role"] == "user"]
    assert ws.leak_problems(t, users) == []


@pytest.mark.asyncio
async def test_the_conversation_never_holds_the_score_or_cap_line(shop, t):
    """The score line after Buy Now and the cap line are method-visible only (the
    learning phase); the actor's conversation never holds either, nor done-page text."""
    env = shop.env()
    task = task_of(env, "test", 1)
    _, steps = await play(env, task)
    assert ws.final_message_pattern(t).fullmatch(steps[-1].messages[-1]["content"])
    bought = steps[-1].fields["conversation"]
    assert bought[-1]["role"] == "assistant" and "buy now" in bought[-1]["content"]
    _never_the_score_or_cap_line(t, bought)
    for step in steps[:-1]:
        _never_the_score_or_cap_line(t, env.actor_messages(step.fields, MEMORY))
    episode = env.episode(task_of(env))
    async with episode:
        await episode.reset()
        for _ in range(protocol.WEBSHOP_MAX_STEPS):
            step = await episode.step(respond("click[nothing]"))
    assert step.messages[-1] == ws.cap_message(t)
    _never_the_score_or_cap_line(t, step.fields["conversation"])


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["learning", "test"])
async def test_the_rollout_sends_the_conversation_and_methods_get_q20(tmp_path, phase):
    """Through the rollout and the gateway: every actor call's recorded request is the
    previous one plus one exchange (a refused call's reply is the placeholder), the
    memory is in the first message only, and methods receive the same Q20 messages as
    before: the pages and actions (no template text), and an empty assistant turn for
    a call without a reply."""
    from evals.skill_bench.agent import rollout as ro
    from evals.skill_bench.tests.fakes import FakeUpstream, Reply
    from evals.skill_bench.tests.rollout_support import (
        SALT,
        RecordingMethod,
        gateway_actor,
        text_reply,
    )

    shop = make_shop(tmp_path / "shop")
    env = shop.env()
    task = env.load_tasks("train")[0]
    calls = 0

    def model(endpoint: str, body: dict[str, Any]) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            return Reply(
                403, {"error": {"code": 403, "message": "Input flagged by moderation"}}
            )
        if calls == 3:
            return text_reply("")
        return text_reply(respond(shopper_action(body["messages"], right=True)))

    method = RecordingMethod(MEMORY)
    async with gateway_actor(tmp_path, FakeUpstream(responder=model)) as (_, actor):
        run = await ro.run_episode(
            env,
            task,
            method=method,
            actor=actor,
            salt=SALT,
            phase=phase,
            idle_timeout_s=5.0 if phase == "learning" else None,
        )
    trajectory = run.trajectory
    assert trajectory.done and run.result.score is not None
    requests = [list(call.request["messages"]) for call in trajectory.calls]
    assert len(requests) == len(trajectory.steps) >= 5
    assert len(requests[0]) == 1 and requests[0][0]["content"].startswith(MEMORY)
    for k, (before, after) in enumerate(zip(requests, requests[1:], strict=False)):
        assert after[: len(before)] == before and len(after) == len(before) + 2
        visible = trajectory.steps[k].assistant["content"]
        expected = env.templates.no_reply_turn if k in (1, 2) else visible
        assert after[len(before)] == {"role": "assistant", "content": expected}
        assert all(MEMORY not in m["content"] for m in after[1:])
    # Q20: what methods receive is unchanged: the raw first page, then per step the
    # model's turn (empty for the refused call) and the page with its actions.
    assert trajectory.steps[1].assistant == {"role": "assistant", "content": ""}
    assert trajectory.steps[2].assistant == {"role": "assistant", "content": ""}
    pages = [m for m in trajectory.messages if m["role"] == "user"]
    assert pages[0]["content"].startswith("WebShop [SEP] Instruction:")
    assert all("Your task is to:" not in m["content"] for m in trajectory.messages)
    assert all("\n\nAvailable actions:\n" in m["content"] for m in pages[:-1])
    if phase == "learning":
        received = [e[1][1] for e in method.events if e[0] == "on_message"]
        assert received == trajectory.messages


@pytest.mark.asyncio
async def test_steps_after_buy_now_are_refused_on_both_sides(shop):
    env = shop.env()
    task = task_of(env, "test", 1)
    episode = env.episode(task)
    async with episode:
        reset = await episode.reset()
        fields = reset.fields
        while True:
            request = env.actor_messages(fields)
            step = await episode.step(respond(shopper_action(request, right=True)))
            fields = step.fields
            if step.done:
                break
        with pytest.raises(RuntimeError, match="over"):
            await episode.step(respond("click[buy now]"))
        prefix = episode.session_prefix
        status, payload = server_module().route(
            shop.service,
            "POST",
            f"/episodes/{prefix}/step",
            lambda: {"action": "click[buy now]"},
        )
        assert status == 409 and "over" in payload["error"]


@pytest.mark.asyncio
async def test_concurrent_episodes_of_one_goal_are_isolated_and_closed(shop):
    """Each episode has its own session prefix; actions in one never move the other;
    close() deletes every session of its prefix, the constructor's orphan included."""
    env = shop.env()
    task = task_of(env)
    one, two = env.episode(task), env.episode(task)
    await asyncio.gather(one.reset(), two.reset())
    assert one.session_prefix != two.session_prefix
    sessions = shop.live_sessions()
    assert len(sessions) == 4  # two goal sessions, two constructor sessions
    step = await one.step(respond("search[red shoes]"))
    assert "'Back to Search'" in step.messages[0]["content"]
    other = await two.step(respond("click[nothing]"))
    assert other.messages[0]["content"].startswith("'Search'")
    await one.close()
    left = shop.live_sessions()
    assert len(left) == 2 and all(s.startswith(str(two.session_prefix)) for s in left)
    await one.close()  # idempotent
    await two.close()
    assert shop.live_sessions() == [] and shop.service.episodes == {}
    with pytest.raises(RuntimeError, match="closed"):
        await two.step(respond("search[x]"))


@pytest.mark.asyncio
async def test_reset_twice_or_step_before_reset_is_refused(shop):
    env = shop.env()
    episode = env.episode(task_of(env))
    with pytest.raises(RuntimeError, match="reset"):
        await episode.step(respond("search[x]"))
    await episode.reset()
    with pytest.raises(RuntimeError, match="once"):
        await episode.reset()
    await episode.close()


@pytest.mark.asyncio
async def test_an_instruction_that_differs_from_the_final_manifest_is_drift(shop):
    env = shop.env()
    task = task_of(env)
    drifted = Task(
        task.benchmark,
        task.split,
        task.raw_id,
        {**task.raw, "instruction_text": "another instruction"},
    )
    episode = env.episode(drifted)
    with pytest.raises(protocol.ProtocolDriftError, match="differs"):
        await episode.reset()
    await episode.close()
    assert shop.live_sessions() == []  # close() still deleted its sessions


@pytest.mark.asyncio
async def test_a_frozen_instruction_ending_in_a_space_is_the_page_s(tmp_path):
    """Upstream strips page texts; an instruction without a price clause can end in a
    space in the goal table (60 human instructions do before their period)."""
    goals = [synthetic_goal(i) for i in range(10)]
    goals[3] = {**goals[3], "instruction_text": "i need dry hair oil "}
    shop = make_shop(tmp_path, goals=goals, goal_count=10, splits={"test": [3]})
    env = shop.env()
    task = Task(
        "webshop",
        "test",
        "fixed_3",
        {"goal_index": 3, "instruction_text": "i need dry hair oil "},
    )
    episode = env.episode(task)
    async with episode:
        reset = await episode.reset()
    assert reset.instruction == "i need dry hair oil"
    assert env.visible_by_design(task) == ["i need dry hair oil"]


@pytest.mark.asyncio
async def test_server_failures_are_infra_and_break_the_episode(shop):
    env = shop.env()
    episode = env.episode(task_of(env))
    async with episode:
        await episode.reset()
        shop.backend.fail_next_step = True
        with pytest.raises(ws.WebShopServerError, match="HTTP 500") as caught:
            await episode.step(respond("search[red shoes]"))
        failure = failures.classify_exception(caught.value)
        assert (failure.kind, failure.code) == (
            failures.FailureKind.INFRA,
            failures.WEBSHOP_SERVER,
        )
        with pytest.raises(ws.WebShopServerError, match="interrupted or failed"):
            await episode.step(respond("search[red shoes]"))


@pytest.mark.asyncio
async def test_a_lost_episode_and_an_unreachable_server_are_infra(shop, tmp_path):
    env = shop.env()
    episode = env.episode(task_of(env))
    await episode.reset()
    shop.service.episodes.clear()  # the server restarted
    with pytest.raises(ws.WebShopServerError, match="HTTP 404"):
        await episode.step(respond("search[x]"))
    await episode.close()

    def down(base_url: str, timeout_s: float) -> httpx.AsyncClient:
        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        return httpx.AsyncClient(
            base_url=base_url, transport=httpx.MockTransport(refuse)
        )

    env = ws.WebShopEnv(server_url=SHOP_URL, client_factory=down)
    episode = env.episode(task_of(env))
    with pytest.raises(ws.WebShopServerError, match="ConnectError") as caught:
        await episode.reset()
    await episode.close()
    assert failures.classify_exception(caught.value).kind == failures.FailureKind.INFRA
    # A refusal of a request the harness should not have sent is a harness error.
    loading = make_shop(tmp_path / "x", manifest_changes={"seed": 7})
    env = loading.env()
    episode = env.episode(task_of(env))
    with pytest.raises(ws.WebShopServerError, match="HTTP 503"):
        await episode.reset()
    await episode.close()


@pytest.mark.asyncio
async def test_a_cancelled_step_waits_for_the_server_before_close(shop):
    """A step cancelled mid-request leaves the server-side state unknown: the episode
    refuses further steps, and close() waits for the request before deleting."""
    gate = asyncio.Event()
    inner = shop.client_factory()

    def slow(base_url: str, timeout_s: float) -> httpx.AsyncClient:
        client = inner(base_url, timeout_s)
        send = client.send

        async def delayed(request: httpx.Request, **kw: Any) -> httpx.Response:
            if request.url.path.endswith("/step"):
                await gate.wait()
            return await send(request, **kw)

        client.send = delayed  # type: ignore[method-assign]
        return client

    env = ws.WebShopEnv(server_url=SHOP_URL, client_factory=slow)
    episode = env.episode(task_of(env))
    await episode.reset()
    running = asyncio.ensure_future(episode.step(respond("search[x]")))
    await asyncio.sleep(0.01)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    closing = asyncio.ensure_future(episode.close())
    await asyncio.sleep(0.01)
    assert not closing.done()  # the shielded request still runs
    gate.set()
    await closing
    assert shop.live_sessions() == []


@pytest.mark.asyncio
async def test_a_cancelled_request_that_fails_later_is_never_an_unread_error(shop):
    """The request a cancelled step left running fails after the step
    was cancelled; its error is read, so asyncio never logs "Future exception was never
    retrieved", and close() still deletes the episode."""
    import gc

    gate = asyncio.Event()
    inner = shop.client_factory()

    def failing(base_url: str, timeout_s: float) -> httpx.AsyncClient:
        client = inner(base_url, timeout_s)
        send = client.send

        async def delayed(request: httpx.Request, **kw: Any) -> httpx.Response:
            if request.url.path.endswith("/step"):
                await gate.wait()
                raise httpx.ReadError("connection reset", request=request)
            return await send(request, **kw)

        client.send = delayed  # type: ignore[method-assign]
        return client

    loop = asyncio.get_running_loop()
    unread: list[dict[str, Any]] = []
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: unread.append(context))
    try:
        env = ws.WebShopEnv(server_url=SHOP_URL, client_factory=failing)
        episode = env.episode(task_of(env))
        await episode.reset()
        running = asyncio.ensure_future(episode.step(respond("search[x]")))
        await asyncio.sleep(0.01)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        closing = asyncio.ensure_future(episode.close())
        await asyncio.sleep(0.01)
        gate.set()
        await closing
        del running, closing, episode
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous)
    assert unread == []
    assert shop.live_sessions() == []


# ---------------------------------------------------------------------------
# The audit host: no server
# ---------------------------------------------------------------------------


def test_the_env_works_without_a_server():
    env = ws.WebShopEnv(client_factory=no_network)
    sizes = dict(protocol.SPLIT_SIZES["webshop"])
    ids: set[str] = set()
    for split in ws.SPLIT_NAMES:
        tasks = env.load_tasks(split)  # type: ignore[arg-type]
        assert len(tasks) == sizes[split]
        assert all(t.raw_id.startswith("fixed_") for t in tasks)
        assert not ids & {t.raw_id for t in tasks}
        ids |= {t.raw_id for t in tasks}
    task = env.load_tasks("test")[0]
    assert env.label_strings(task) == []
    assert env.answers(task) == [task.raw["asin"]]
    # The first page shows the instruction with its price limit, which only a final
    # manifest holds (a provisional one has the instruction without it).
    final = env.manifest_status() == ws.MANIFEST_FINAL
    shown = task.raw["instruction_text"] if final else task.raw["instruction"]
    assert env.visible_by_design(task) == [shown]
    assert env.episode(task) is not None  # building an episode sends nothing
    score = env.score(task, None, {"purchased": True, "reward": 0.5, "steps": 3})
    assert (score.primary, score.metrics["success"]) == (0.5, 0.0)
    with pytest.raises(ValueError, match="unknown WebShop split"):
        env.load_tasks("dev")  # type: ignore[arg-type]
    assert env.manifest_status() in (ws.MANIFEST_PROVISIONAL, ws.MANIFEST_FINAL)


@pytest.mark.parametrize(
    "text",
    [
        # A seller's own words, inside a longer part.
        "Machine washable. Thank you for shopping with us! Please contact us with any "
        "question.",
        "Best price upper arm blood pressure monitor",
        "Reward Details: earn points with every purchase",
        "Your score (min 0.0, max 1.0) on our survey matters",
        "Paste it in your MTurk interface",
        # A bullet that is exactly a done-page header word, not followed by its label.
        "Thank you for shopping with us!",
        "Purchased",
        "Target",
    ],
)
def test_seller_text_is_not_done_page_text(t, text):
    sep = t.observation_separator
    for page in (
        f"'Back to Search'{sep}'< Prev'{sep}'{text}'{sep}'Buy Now'",
        f"WebShop{sep}Instruction:{sep}i need shoes{sep}{text}{sep}Next >",
    ):
        message = ws.observation_message(t, page, ["click[buy now]"])
        assert ws.leak_problems(t, [message]) == [], page


def test_page_parts_drop_quotes_whitespace_and_empty_parts(t):
    sep = t.observation_separator
    assert ws.page_parts(t, f"'a'{sep}  {sep}'b c'\n\nAvailable actions:\n'x',") == [
        "a",
        "b c",
        "Available actions:",
        "'x',",
    ]
    assert ws.page_parts(t, "''") == [] and ws.page_parts(t, "'") == ["'"]


@pytest.mark.asyncio
async def test_preflight_refuses_a_provisional_manifest_before_any_request(
    monkeypatch,
):
    """Whatever the repository's manifest says, a provisional one is refused before any
    request."""
    env = ws.WebShopEnv(client_factory=no_network)
    monkeypatch.setattr(env, "manifest_status", lambda: ws.MANIFEST_PROVISIONAL)
    with pytest.raises(ws.WebShopEnvError, match="provisional"):
        await env.preflight()


@pytest.mark.parametrize(
    ("private", "primary", "success", "within", "answered"),
    [
        ({"purchased": True, "reward": 1.0, "steps": 15}, 1.0, 1.0, 1.0, True),
        ({"purchased": True, "reward": 1.0, "steps": 16}, 1.0, 1.0, 0.0, True),
        ({"purchased": True, "reward": 0.99, "steps": 2}, 0.99, 0.0, 0.0, True),
        ({"purchased": True, "reward": 0, "steps": 2}, 0.0, 0.0, 0.0, True),
        ({"purchased": False, "reward": None, "steps": 15}, 0.0, 0.0, 0.0, False),
    ],
)
def test_score(private, primary, success, within, answered):
    env = ws.WebShopEnv(client_factory=no_network)
    task = task_of(env)
    with_best = Task(
        task.benchmark, task.split, task.raw_id, {**task.raw, "best_reward": 0.75}
    )
    score = env.score(with_best, "ignored", private)
    assert (score.primary, score.answered) == (primary, answered)
    assert dict(score.metrics) == {
        "success": success,
        "success_within_15": within,
        "best_reward": 0.75,
    }


@pytest.mark.parametrize(
    "private",
    [
        {"purchased": True, "reward": 1.5, "steps": 2},
        {"purchased": True, "reward": None, "steps": 2},
        {"purchased": "yes", "reward": 1.0, "steps": 2},
        {"purchased": True, "reward": 1.0},
    ],
)
def test_score_refuses_a_malformed_private(private):
    env = ws.WebShopEnv(client_factory=no_network)
    with pytest.raises(ValueError):
        env.score(task_of(env), None, private)


def test_the_frozen_splits_match_release_contract():
    """Ranges, sizes, no gold ASIN or normalized instruction shared across splits,
    one goal per ASIN within train and within val, ids ``fixed_<idx>``."""
    from evals.skill_bench.tools import prepare_webshop

    assert prepare_webshop.check(ROOT / ws.SPLIT_DIR) == []
    manifest = ws.load_split_manifest()
    assert set(manifest["files"]) == {"train.json", "val.json", "test.json"}
    assert manifest["counts"] == {
        "train": 40,
        "val": 20,
        "test": 140,
    }
    test = ws.load_split("test")
    assert all(100 <= row["goal_index"] < 500 for row in test)


def test_split_files_sha256_names_every_manifest_file():
    hashes = ws.split_files_sha256()
    assert set(hashes) == {
        "train.json",
        "val.json",
        "test.json",
        "split_manifest.json",
        "source.json",
    }


def test_a_final_manifest_needs_every_pinned_field(tmp_path, monkeypatch):
    # Isolate schema validation; checksum rejection is covered by the loader tests.
    monkeypatch.setattr(protocol, "verify_frozen_assets", lambda *_args: [])
    rows = [{"id": "fixed_101", "goal_index": 101, "asin": "B0", "instruction": "i"}]
    directory = tmp_path / ws.SPLIT_DIR
    directory.mkdir(parents=True)
    (directory / "test.json").write_text(json.dumps(rows))
    (directory / "split_manifest.json").write_text(json.dumps({"status": "final"}))
    with pytest.raises(ws.WebShopEnvError, match="final manifest"):
        ws.load_split("test", tmp_path)
    (directory / "split_manifest.json").write_text(json.dumps({"status": "draft"}))
    with pytest.raises(ws.WebShopEnvError, match="no valid status"):
        ws.load_split("test", tmp_path)
    (directory / "split_manifest.json").write_text(
        json.dumps({"status": "provisional"})
    )
    (directory / "test.json").write_text(json.dumps([{**rows[0], "id": "101"}]))
    with pytest.raises(ws.WebShopEnvError, match="fixed_<goal_index>"):
        ws.load_split("test", tmp_path)


# ---------------------------------------------------------------------------
# Preflight against a ready server
# ---------------------------------------------------------------------------


def _ready_info(shop: Shop) -> dict[str, Any]:
    return shop.service.readiness()[1]["info"]


def _expected_from(shop: Shop) -> dict[str, Any]:
    info = _ready_info(shop)
    return {
        "api_version": ws.SERVER_API_VERSION,
        "mode": "serve",
        "seed": protocol.SEED,
        "goal_count": protocol.WEBSHOP_GOAL_COUNT,
        "price_table_sha256": info["price_table_sha256"],
        "manifest_files": info["manifest"]["files"],
        "upstream_commit": protocol.WEBSHOP_SERVER["upstream_commit"],
        "spacy_model": protocol.WEBSHOP_SERVER["spacy_model"],
        "fuzzy_backend": protocol.WEBSHOP_SERVER["fuzzy_backend"],
        "data": {n: p["sha256"] for n, p in protocol.WEBSHOP_DATA["files"].items()},
        "index_sha256": INDEX_SHA256,
        "context": ws.context_digest(),
    }


def test_expected_server_is_the_protocol(shop):
    env = shop.env()
    expected = env.expected_server()
    assert expected["seed"] == protocol.SEED and expected["mode"] == "serve"
    assert expected["upstream_commit"] == protocol.WEBSHOP_SERVER["upstream_commit"]
    assert expected["manifest_files"] == ws.split_files_sha256()
    assert expected["context"] == ws.context_digest()
    assert expected["index_sha256"] == json.loads(
        (ROOT / ws.SOURCE_FILE).read_text()
    ).get("index_sha256")
    assert set(expected) == set(_expected_from(shop))
    assert ws.server_problems(_ready_info(shop), _expected_from(shop)) == []


@pytest.mark.parametrize(
    ("change", "named"),
    [
        (lambda i: i.update(seed=7), "seed"),
        (lambda i: i.update(price_table_sha256="0" * 64), "price_table_sha256"),
        (lambda i: i["manifest"]["files"].update({"test.json": "0"}), "manifest_files"),
        (lambda i: i["build"].update(upstream_commit="abc"), "upstream_commit"),
        (lambda i: i["build"]["data"].update({"items_shuffle.json": "0"}), "data"),
        (lambda i: i.update(fuzzy_backend="Levenshtein"), "fuzzy_backend"),
        (lambda i: i.update(spacy_model="en_core_web_sm==3.7.0"), "spacy_model"),
        (lambda i: i.update(mode="prepare"), "mode"),
        (lambda i: i.update(api_version=2), "api_version"),
        # A rebuilt index ranks products differently.
        (lambda i: i["build"].update(index_sha256="f" * 64), "index_sha256"),
        (lambda i: i["build"].pop("index_sha256"), "index_sha256"),
        # An image built from another server directory (a stale server.py).
        (lambda i: i["build"].update(context="0" * 64), "context"),
    ],
)
def test_server_problems_name_each_difference(shop, change, named):
    info = json.loads(json.dumps(_ready_info(shop)))
    change(info)
    problems = ws.server_problems(info, _expected_from(shop))
    assert len(problems) == 1 and problems[0].startswith(named)


@pytest.mark.asyncio
async def test_preflight_against_a_ready_server(shop, monkeypatch):
    env = shop.env()
    monkeypatch.setattr(env, "manifest_status", lambda: ws.MANIFEST_FINAL)
    monkeypatch.setattr(env, "expected_server", lambda: _expected_from(shop))

    async def image_id(name: str = ws.CONTAINER_NAME) -> str:
        return "sha256:feed"

    async def state(name: str = ws.CONTAINER_NAME) -> dict[str, Any]:
        return {
            "ours": True,
            "status": "running",
            "restart_count": 2,
            "restart_policy": "unless-stopped",
            "image": "sha256:feed",
        }

    monkeypatch.setattr(ws, "container_image_id", image_id)
    monkeypatch.setattr(ws, "container_state", state)
    record = await env.preflight()
    assert record["sandbox_image_id"] == "sha256:feed"
    # Recorded, not blocked on: how often Docker brought the server back.
    assert record["container"] == {
        "status": "running",
        "restart_count": 2,
        "restart_policy": "unless-stopped",
    }
    assert record["server"]["seed"] == protocol.SEED
    assert (
        record["server"]["build"]["upstream_commit"]
        == (protocol.WEBSHOP_SERVER["upstream_commit"])
    )
    # Another manifest on the server: drift, never a silent start.
    monkeypatch.setattr(
        env,
        "expected_server",
        lambda: {**_expected_from(shop), "manifest_files": {"test.json": "0"}},
    )
    with pytest.raises(protocol.ProtocolDriftError, match="manifest_files"):
        await env.preflight()


@pytest.mark.asyncio
async def test_preflight_refuses_a_server_that_refused_its_manifest(
    tmp_path, monkeypatch
):
    """A server whose goals or price table differ from the frozen manifest answers
    ``failed`` and preflight refuses it: a person fixes it (never INFRA, which would
    re-run the cell against the same server)."""
    shop = make_shop(tmp_path, manifest_changes={"prices_sha256": "0" * 64})
    assert shop.service.state == "failed"
    env = shop.env()
    monkeypatch.setattr(env, "manifest_status", lambda: ws.MANIFEST_FINAL)
    monkeypatch.setattr(env, "expected_server", lambda: _expected_from(shop))
    with pytest.raises(ws.WebShopEnvError, match="'failed'.*price table") as refused:
        await env.preflight()
    assert classify_exception(refused.value).kind is FailureKind.HARNESS


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply", "error", "kind"),
    [
        # Unreachable or still loading is transient (INFRA).
        (None, ws.WebShopServerError, FailureKind.INFRA),
        ((503, {"state": "loading"}), ws.WebShopServerError, FailureKind.INFRA),
        # Not a serving server: a person.
        ((503, {"state": "prepare"}), ws.WebShopEnvError, FailureKind.HARNESS),
        ((503, {"state": "failed", "problems": ["x"]}), ws.WebShopEnvError, None),
    ],
)
async def test_preflight_tells_a_transient_server_from_a_broken_one(
    monkeypatch, reply, error, kind
):
    def handle(request: httpx.Request) -> httpx.Response:
        if reply is None:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(reply[0], json=reply[1])

    def factory(base_url: str, timeout_s: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=base_url, transport=httpx.MockTransport(handle)
        )

    env = ws.WebShopEnv(server_url=SHOP_URL, client_factory=factory)
    monkeypatch.setattr(env, "manifest_status", lambda: ws.MANIFEST_FINAL)
    monkeypatch.setattr(env, "expected_server", lambda: {"index_sha256": INDEX_SHA256})
    with pytest.raises(error) as raised:
        await env.preflight()
    if kind is not None:
        assert classify_exception(raised.value).kind is kind


@pytest.mark.asyncio
async def test_preflight_refuses_a_final_manifest_without_its_index(monkeypatch):
    env = ws.WebShopEnv(client_factory=no_network)
    monkeypatch.setattr(env, "manifest_status", lambda: ws.MANIFEST_FINAL)
    for index in (None, "", "F" * 64, "0" * 63):
        monkeypatch.setattr(
            env,
            "expected_server",
            lambda index=index: {"index_sha256": index},
        )
        with pytest.raises(ws.WebShopEnvError, match="no index_sha256"):
            await env.preflight()


@pytest.mark.asyncio
async def test_wait_ready_polls_until_ready_and_raises_on_failed(tmp_path):
    states = iter(["loading", "loading", "ready"])

    def reply(request: httpx.Request) -> httpx.Response:
        state = next(states)
        return httpx.Response(200 if state == "ready" else 503, json={"state": state})

    def factory(base_url: str, timeout_s: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=base_url, transport=httpx.MockTransport(reply)
        )

    payload = await ws.wait_ready(SHOP_URL, timeout_s=5, poll_s=0.0, factory=factory)
    assert payload["state"] == "ready"
    failed = make_shop(tmp_path, manifest_changes={"seed": 1})
    with pytest.raises(ws.WebShopServerError, match="failed"):
        await ws.wait_ready(
            SHOP_URL, timeout_s=5, poll_s=0.0, factory=failed.client_factory()
        )

    def loading(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"state": "loading"})

    def slow(base_url: str, timeout_s: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=base_url, transport=httpx.MockTransport(loading)
        )

    with pytest.raises(ws.WebShopServerError, match="still 'loading'"):
        await ws.wait_ready(SHOP_URL, timeout_s=0.0, poll_s=0.0, factory=slow)


# ---------------------------------------------------------------------------
# The server image's pins
# ---------------------------------------------------------------------------

SERVER = ROOT / ws.SERVER_DIR


def test_the_dockerfile_is_built_from_the_protocol():
    text = (SERVER / "Dockerfile").read_text()
    assert ws.dockerfile_base_image() == protocol.WEBSHOP_SERVER["base_image"]
    assert f"ARG WEBSHOP_COMMIT={protocol.WEBSHOP_SERVER['upstream_commit']}" in text
    assert '"--seed", "42"' in text and protocol.SEED == 42
    assert "--require-hashes" in text and "--no-deps" in text
    # Only the hash-locked allowlist: never upstream's requirements.txt or setup.sh.
    assert "-r /opt/skill-bench/requirements.txt" in text
    assert "/opt/webshop/requirements" not in text and "setup.sh" not in text.replace(
        "setup.sh:29", ""
    )
    assert protocol.WEBSHOP_SERVER["java"] in text
    argv = ws.build_argv()
    assert argv[:4] == ["docker", "build", "--platform", "linux/amd64"]
    assert argv[-2] == ws.image_name() and argv[-1] == str(SERVER)
    # The image records the context it was built from (preflight compares it).
    at = argv.index("--build-arg")
    assert argv[at + 1] == f"SKILL_BENCH_CONTEXT={ws.context_digest()}"
    assert ws.image_name().endswith(":" + ws.context_digest()[:16])
    assert server_module().CONTEXT_ENV == ws.CONTEXT_BUILD_ARG
    # Declared just before the record's own layer: a new context rebuilds that only.
    steps = dockerfile_instructions()
    at = steps.index(f"ARG {ws.CONTEXT_BUILD_ARG}")
    assert steps[at + 1] == (
        "RUN python /opt/skill-bench/server.py write-build-record "
        "--data-dir /opt/webshop-data"
    )
    assert not [x for x in steps[at + 2 :] if x.startswith(("RUN", "ARG", "COPY"))]


def dockerfile_instructions(path: Path | None = None) -> list[str]:
    """The Dockerfile's instructions, one string each: comments dropped, continuation
    lines joined, runs of whitespace collapsed."""
    text = (path or SERVER / "Dockerfile").read_text()
    joined = re.sub(r"\\\n", " ", text)
    return [
        " ".join(line.split())
        for line in joined.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


#: The image's layers in order: the product files depend only on data.json and
#: data_index.py; the index on those, the
#: packages and the upstream commit; server.py and what changes often come last.
DOCKER_LAYER_ORDER = (
    "FROM ",
    "ENV ",
    "COPY data.json data_index.py /opt/skill-bench/",
    "RUN python /opt/skill-bench/data_index.py fetch-data --out /opt/webshop-data",
    "RUN printf '%s\\n'",
    "COPY requirements.txt /opt/skill-bench/requirements.txt",
    "RUN pip install",
    "ARG WEBSHOP_COMMIT=",
    "RUN git clone https://github.com/princeton-nlp/WebShop.git /opt/webshop",
    "RUN python /opt/skill-bench/data_index.py build-index --data-dir /opt/webshop-data",
    "COPY server.py /opt/skill-bench/",
    "RUN python /opt/skill-bench/server.py check-install",
    f"ARG {ws.CONTEXT_BUILD_ARG}",
    "RUN python /opt/skill-bench/server.py write-build-record",
    "WORKDIR /opt/webshop",
    "EXPOSE 8000",
    'ENTRYPOINT ["python", "/opt/skill-bench/server.py"]',
    'CMD ["serve",',
)


def test_the_dockerfile_layers_put_the_data_and_index_before_the_server():
    steps = dockerfile_instructions()
    assert len(steps) == len(DOCKER_LAYER_ORDER), steps
    for step, start in zip(steps, DOCKER_LAYER_ORDER, strict=True):
        assert step.startswith(start), (step, start)
    fetch = next(i for i, x in enumerate(steps) if "fetch-data" in x)
    index = next(i for i, x in enumerate(steps) if "build-index" in x)
    server = steps.index("COPY server.py /opt/skill-bench/")
    assert fetch < index < server
    # Nothing before the server's layer copies or runs server.py.
    assert not [x for x in steps[:server] if "server.py" in x]
    # Before the data is fetched only its pins and its script are in the image.
    copied = [x.split()[1:-1] for x in steps[:fetch] if x.startswith("COPY")]
    assert copied == [["data.json", "data_index.py"]]
    # The data directory: data_index.py's default, the ENV, every step's argument, and
    # outside upstream's checkout (fetched before the clone).
    env = next(x for x in steps if x.startswith("ENV "))
    assert "WEBSHOP_DATA_DIR=/opt/webshop-data" in env.split()
    assert "WEBSHOP_DIR=/opt/webshop" in env.split()
    assert all("/opt/webshop/data" not in x for x in steps)
    # Every file the image copies is in the context (whose digest is the image's tag).
    context = {p.name for p in SERVER.iterdir() if p.is_file()}
    sources = {src for x in steps if x.startswith("COPY ") for src in x.split()[1:-1]}
    assert sources <= context and context - sources == {"Dockerfile", "requirements.in"}


#: The base image's own apt snapshot: ``python:3.9-bullseye@sha256:3dcaf39b...``'s
#: ``/etc/apt/sources.list`` names snapshot.debian.org at this date for each of its three
#: suites, commented out.
BASE_IMAGE_APT_SNAPSHOT = "20250721T000000Z"
BASE_IMAGE_APT_SUITES = {
    "bullseye": "debian",
    "bullseye-security": "debian-security",
    "bullseye-updates": "debian",
}


def test_apt_reads_the_base_images_own_snapshot(tmp_path):
    """bullseye's LTS has ended: deb.debian.org may still list bullseye-security without
    serving its packages, and archive.debian.org may not hold it yet. The image's apt
    reads snapshot.debian.org at
    the base image's own date instead, for the three suites the base image uses, and
    nothing else; the sources are written in the same layer as the install, before
    ``apt-get update``. The step's ``printf`` is run here as the image's shell runs it."""
    import shlex
    import subprocess

    steps = dockerfile_instructions()
    [apt] = [x for x in steps if "apt-get" in x]
    assert apt == next(x for x in steps if x.startswith("RUN printf"))
    assert "deb.debian.org" not in apt and "security.debian.org" not in apt
    assert "archive.debian.org" not in apt
    write, _, rest = apt[len("RUN ") :].partition(" > /etc/apt/sources.list ")
    assert rest.startswith("&& apt-get -o Acquire::Retries=5 update ")
    assert (
        f"install -y --no-install-recommends {protocol.WEBSHOP_SERVER['java']} git"
        in (rest)
    )
    assert rest.endswith("&& rm -rf /var/lib/apt/lists/*")
    assert shlex.split(write)[0] == "printf"
    out = subprocess.run(
        ["sh", "-c", write], check=True, capture_output=True, text=True
    ).stdout
    lines = out.splitlines()
    pattern = re.compile(
        r"deb \[check-valid-until=no\] http://snapshot\.debian\.org/archive/"
        r"(debian|debian-security)/(\d{8}T\d{6}Z) (\S+) main"
    )
    found = {}
    for line in lines:
        match = pattern.fullmatch(line)
        assert match, line
        archive, date, suite = match.groups()
        assert date == BASE_IMAGE_APT_SNAPSHOT, line
        found[suite] = archive
    assert found == BASE_IMAGE_APT_SUITES and len(lines) == len(found)


def _layer_keys(context: Path) -> list[tuple[str, str]]:
    """A model of Docker's layer cache (the classic builder's, which BuildKit refines but
    never loosens): each layer's key is the previous key plus its instruction, plus for
    ``COPY`` the bytes of what it copies, plus for ``RUN`` every build argument declared
    so far (an argument's value is in the environment of every later ``RUN``). The build
    passes ``SKILL_BENCH_CONTEXT`` = the context digest, as ``build_argv`` does."""
    build_args = {ws.CONTEXT_BUILD_ARG: ws.context_digest(context.parents[2])}
    declared: dict[str, str] = {}
    key = ""
    out = []
    for step in dockerfile_instructions(context / "Dockerfile"):
        material = step
        if step.startswith("COPY "):
            material += "".join(
                hashlib.sha256((context / src).read_bytes()).hexdigest()
                for src in step.split()[1:-1]
            )
        elif step.startswith("ARG "):
            name, _, default = step[4:].partition("=")
            declared[name] = build_args.get(name, default)
        elif step.startswith("RUN "):
            material += json.dumps(declared, sort_keys=True)
        key = hashlib.sha256((key + "\0" + material).encode()).hexdigest()
        out.append((step, key))
    return out


@pytest.mark.parametrize(
    ("changed", "first_rebuilt"),
    [
        ("server.py", "COPY server.py"),
        ("requirements.txt", "COPY requirements.txt"),
        ("data.json", "COPY data.json data_index.py"),
        ("data_index.py", "COPY data.json data_index.py"),
        # Only the build record's layer (through the context digest).
        ("requirements.in", "RUN python /opt/skill-bench/server.py write-build-record"),
    ],
)
def test_a_change_rebuilds_only_the_layers_from_its_own(
    tmp_path, changed, first_rebuilt
):
    """A change to server.py never fetches the 5.7 GB of product
    files again or rebuilds the index; a change to the data pins or the data script
    does; a change of the packages rebuilds the index but keeps the data."""
    import shutil

    copy = tmp_path / ws.SERVER_DIR
    shutil.copytree(SERVER, copy)
    before = _layer_keys(copy)
    with open(copy / changed, "a", encoding="utf-8") as handle:
        handle.write("\n# changed\n")
    after = _layer_keys(copy)
    assert [s for s, _ in before] == [s for s, _ in after]
    first = next(i for i, (b, a) in enumerate(zip(before, after)) if b[1] != a[1])
    assert before[first][0].startswith(first_rebuilt), before[first][0]
    rebuilt = [s for s, _ in before[first:]]
    fetch = any("fetch-data" in s for s in rebuilt)
    index = any("build-index" in s for s in rebuilt)
    assert fetch is (changed in ("data.json", "data_index.py"))
    assert index is (changed in ("data.json", "data_index.py", "requirements.txt"))
    # Whatever changed, the image's tag and its build record's context change too.
    assert ws.image_name(tmp_path) != ws.image_name()
    assert any("write-build-record" in s for s in rebuilt)


def test_build_refuses_another_base_image(tmp_path):
    root = tmp_path
    (root / ws.SERVER_DIR).mkdir(parents=True)
    (root / ws.SERVER_DIR / "Dockerfile").write_text("FROM python:3.9\n")
    with pytest.raises(protocol.ProtocolDriftError):
        ws.build_argv(root)


def test_data_pins_are_the_protocols():
    pins = json.loads((SERVER / "data.json").read_text())
    assert pins["hf_repo"] == protocol.WEBSHOP_DATA["hf_repo"]
    assert pins["hf_revision"] == protocol.WEBSHOP_DATA["hf_revision"]
    assert pins["files"] == protocol.WEBSHOP_DATA["files"]


def test_the_requirements_are_a_hash_locked_allowlist():
    """Every pin carries hashes, no python-Levenshtein, the spaCy model
    wheel pinned, thefuzz 0.19 (difflib without Levenshtein), pyserini 0.17."""
    lock = (SERVER / "requirements.txt").read_text()
    pins = re.findall(r"^([A-Za-z0-9_.-]+)(?:==| @ )(\S+)", lock, re.MULTILINE)
    assert pins
    names = {name.lower().replace("_", "-") for name, _ in pins}
    assert "python-levenshtein" not in names and "levenshtein" not in names
    assert {"thefuzz", "spacy", "pyserini", "gym", "en-core-web-sm"} <= names
    blocks = re.split(r"\n(?=[A-Za-z0-9_.-]+(?:==| @ ))", lock.split("\n\n", 1)[-1])
    for block in blocks:
        if re.match(r"[A-Za-z0-9_.-]+(?:==| @ )", block):
            assert "--hash=sha256:" in block, block.splitlines()[0]
    versions = dict(pins)
    assert versions.get("pyserini") == protocol.WEBSHOP_SERVER["pyserini"]
    assert versions.get("thefuzz") == "0.19.0"
    assert versions.get("spacy") == "3.3.0"
    assert "en_core_web_sm-3.3.0" in lock
    allow = (SERVER / "requirements.in").read_text()
    # The requirement lines, comments dropped (the file starts with its comment header).
    wanted = [x.split("#")[0].strip().lower() for x in allow.splitlines()]
    assert wanted and not [x for x in wanted if x.startswith(("python-lev", "lev"))]
    # spaCy 3.3.0 needs pydantic < 1.9, and pydantic 1.8 breaks on typing_extensions 4.6
    # and later (importing spaCy raises in the image's install check).
    assert versions["pydantic"].startswith("1.8.")
    major, minor = (int(x) for x in versions["typing-extensions"].split(".")[:2])
    assert (major, minor) < (4, 6), versions["typing-extensions"]
    assert re.search(r"^typing-extensions<4\.6$", allow, re.MULTILINE)


@pytest.mark.parametrize("name", ["server.py", "data_index.py"])
def test_the_server_is_python_39_standard_library_code(name):
    """Both image scripts are Python 3.9 and import the standard library only (the
    server also its sibling data_index.py; data_index.py runs before anything is
    installed)."""
    source = (SERVER / name).read_text()
    ast.parse(source, feature_version=(3, 9))
    tree = ast.parse(source)
    top = {
        (node.module if isinstance(node, ast.ImportFrom) else alias.name).split(".")[0]
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    import sys

    sibling = {"data_index"} if name == "server.py" else set()
    assert top <= set(sys.stdlib_module_names) | {"__future__"} | sibling, top
    assert server_module().API_VERSION == ws.SERVER_API_VERSION
    assert server_module().MANIFEST_FIELDS == ws.MANIFEST_FIELDS
    assert (
        server_module().SPECIAL_SEARCH.pattern
        == ws.WebShopTemplates.load().special_search_pattern
    )


def test_run_argv_is_loopback_labelled_capped_and_read_only(tmp_path):
    argv = ws.run_argv(port=18560)
    text = " ".join(argv)
    assert "-p 127.0.0.1:18560:8000" in text
    assert f"--label {ws.CONTAINER_LABEL_KEY}={ws.CONTAINER_LABEL_VALUE}" in text
    assert "--memory 24g --memory-swap 24g" in text and "--cpus 2" in text
    assert "target=/manifest,readonly" in text and "--manifest-dir /manifest" in text
    assert f"--seed {protocol.SEED}" in text
    # A crashed server (an OOM kill at its cap) comes back by itself; so does it after a
    # Docker daemon restart or a reboot.
    assert "--restart unless-stopped" in text and "on-failure" not in text
    prepare = " ".join(ws.run_argv(port=18561, prepare=True))
    assert "--prepare" in prepare and "/manifest" not in prepare
    assert ws.PREPARE_CONTAINER_NAME in prepare
    assert "prune" not in text and "opensearch" not in text


@pytest.mark.asyncio
async def test_containers_are_stopped_only_with_the_label(monkeypatch):
    calls: list[tuple[str, ...]] = []
    labels = {ws.CONTAINER_NAME: "webshop", "opensearch": ""}

    async def docker(*argv: str, timeout_s: float = 60.0) -> tuple[int, str]:
        calls.append(argv)
        name = argv[-1]
        if argv[1] == "inspect":
            if name not in labels:
                return 1, ""
            image = " sha256:abc" if "{{.Image}}" in argv[3] else ""
            return 0, labels[name] + image
        return 0, ""

    monkeypatch.setattr(ws, "_docker", docker)
    assert await ws.stop_container("opensearch") is False
    assert not any(c[1] == "rm" for c in calls)
    assert await ws.stop_container("absent") is False
    assert await ws.stop_container() is True
    assert ("docker", "rm", "-f", ws.CONTAINER_NAME) in calls
    assert await ws.container_image_id() == "sha256:abc"
    assert await ws.container_image_id("opensearch") is None


class FakeDocker:
    """The docker CLI as :mod:`envs.webshop` calls it, over a dict of containers
    (name -> label, status, restart count, policy, image)."""

    def __init__(self, **containers: tuple[str, str, int, str, str]) -> None:
        self.containers = dict(containers)
        self.calls: list[tuple[str, ...]] = []

    async def __call__(
        self, *argv: str, timeout_s: float = 60.0, capture: bool = True
    ) -> tuple[int, str]:
        self.calls.append(argv)
        verb, name = argv[1], argv[-1]
        if verb == "inspect":
            if name not in self.containers:
                return 1, ""
            label, status, restarts, policy, image = self.containers[name]
            if argv[3] == ws._STATE_FORMAT:
                return 0, f"{label}|{status}|{restarts}|{policy}|{image}"
            return 0, label
        if verb == "rm":
            self.containers.pop(name, None)
            return 0, ""
        if verb == "run":
            name = argv[argv.index("--name") + 1]
            if name in self.containers:
                return 125, ""  # Conflict. The container name is already in use
            policy = argv[argv.index("--restart") + 1]
            self.containers[name] = ("webshop", "running", 0, policy, "sha256:new")
            return 0, "c0ffee"
        return 1, ""


@pytest.mark.asyncio
async def test_start_runs_keeps_or_replaces_the_container(monkeypatch):
    """After a reboot a container started under ``on-failure:3`` stays exited and holds
    its name, so a plain ``docker run`` would fail. ``--start``: none -> run; ours running -> left as it is; ours exited -> removed and run again with this
    checkout's options and restart policy; another container of that name -> refused,
    untouched."""
    docker = FakeDocker()
    monkeypatch.setattr(ws, "_docker", docker)
    running, what = await ws.start_container(port=18560)
    assert running and what == f"started {ws.CONTAINER_NAME} (c0ffee)"
    assert docker.containers[ws.CONTAINER_NAME][3] == "unless-stopped"
    docker.calls.clear()
    running, what = await ws.start_container(port=18560)
    assert running and "already runs" in what and "left as it is" in what
    assert [c[1] for c in docker.calls] == ["inspect"]
    # After a reboot under the former policy: exited, holding the name.
    docker.containers[ws.CONTAINER_NAME] = (
        "webshop",
        "exited",
        3,
        "on-failure",
        "sha256:old",
    )
    docker.calls.clear()
    running, what = await ws.start_container(port=18560)
    assert running
    assert what.startswith(
        f"removed the exited container {ws.CONTAINER_NAME} (restart policy on-failure); "
    )
    assert [c[1] for c in docker.calls] == ["inspect", "inspect", "rm", "run"]
    assert docker.containers[ws.CONTAINER_NAME][1:4] == ("running", 0, "unless-stopped")
    # Someone else's container under the name is never touched.
    docker.containers[ws.CONTAINER_NAME] = ("", "exited", 0, "no", "sha256:x")
    docker.calls.clear()
    running, what = await ws.start_container(port=18560)
    assert not running and "this harness did not start" in what
    assert [c[1] for c in docker.calls] == ["inspect"]
    # The prepare container likewise, under its own name.
    running, what = await ws.start_container(port=18561, prepare=True)
    assert running and ws.PREPARE_CONTAINER_NAME in what


@pytest.mark.asyncio
async def test_container_state_reads_docker(monkeypatch):
    docker = FakeDocker(
        **{ws.CONTAINER_NAME: ("webshop", "running", 4, "unless-stopped", "sha256:a")}
    )
    monkeypatch.setattr(ws, "_docker", docker)
    assert await ws.container_state() == {
        "ours": True,
        "status": "running",
        "restart_count": 4,
        "restart_policy": "unless-stopped",
        "image": "sha256:a",
    }
    assert await ws.container_state("absent") is None


def test_the_start_cli_relaunches_after_a_reboot(monkeypatch, capsys):
    docker = FakeDocker(
        **{ws.CONTAINER_NAME: ("webshop", "exited", 3, "on-failure", "sha256:old")}
    )
    monkeypatch.setattr(ws, "_docker", docker)
    assert ws._main(["--start", "--port", "18560"]) == 0
    assert "removed the exited container" in capsys.readouterr().out
    assert ws._main(["--start", "--port", "18560"]) == 0
    assert "already runs" in capsys.readouterr().out
    docker.containers[ws.CONTAINER_NAME] = ("", "running", 0, "no", "sha256:x")
    assert ws._main(["--start", "--port", "18560"]) == 1


@pytest.mark.asyncio
async def test_docker_without_the_cli_is_no_answer(monkeypatch):
    async def missing(*argv: Any, **kw: Any) -> Any:
        raise FileNotFoundError("docker")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", missing)
    assert await ws._docker("docker", "ps") == (-1, "")
    assert await ws.container_image_id() is None


def test_image_name_follows_the_docker_context(tmp_path):
    import shutil

    copy = tmp_path / ws.SERVER_DIR
    shutil.copytree(SERVER, copy)
    assert ws.image_name(tmp_path) == ws.image_name()
    assert ws.image_name().startswith("skill-bench-webshop:")
    (copy / "server.py").write_text((copy / "server.py").read_text() + "\n# x\n")
    assert ws.image_name(tmp_path) != ws.image_name()


def test_print_argv(capsys):
    assert ws._main(["--print-argv", "--port", "18570"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("docker build") and "127.0.0.1:18570:8000" in out[1]
    assert "--prepare" in out[2]


def test_the_env_registers_everywhere():
    from evals.skill_bench import report, run, schedule
    from evals.skill_bench.agent import rollout

    assert "webshop" in protocol.BENCHMARKS
    assert isinstance(run.default_env("webshop"), ws.WebShopEnv)
    assert "webshop" in rollout._DRIVERS and "webshop" in report.METRICS
    assert schedule.BENCHMARK_ORDER[-1] == "webshop"


def test_the_rollout_driver_caps_at_fifteen_steps():
    from evals.skill_bench.agent import rollout
    from evals.skill_bench.envs.base import Reset

    cls = rollout._DRIVERS["webshop"]
    assert cls is not rollout._DRIVERS["alfworld"]
    env = ws.WebShopEnv(client_factory=no_network)
    driver = cls(env, Reset(instruction="i", messages=(), fields={}))
    assert driver.max_calls == protocol.WEBSHOP_MAX_STEPS == 15  # never ALFWorld's 50
    assert set(driver.record_fields) >= {
        "action",
        "format_error",
        "action_valid",
        "accepted",
        "truncated",
    }
    assistant, action = driver.without_reply()  # type: ignore[misc]
    assert assistant == {"role": "assistant", "content": ""}
    assert action == {ws.NO_REPLY_KEY: True}
    short = ws.WebShopEnv(client_factory=no_network, max_steps=5)
    assert cls(short, Reset(instruction="i", messages=(), fields={})).max_calls == 5
