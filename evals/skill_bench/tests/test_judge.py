"""The shared judge J: prompt faithfulness, parsing, isolation."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest

from evals.skill_bench import protocol
from evals.skill_bench.agent.actor import ActorCallError
from evals.skill_bench.infra.failures import FailureKind
from evals.skill_bench.methods import judge as J
from evals.skill_bench.methods.base import (
    CellContext,
    TaskView,
    TrajectoryView,
)
from evals.skill_bench.tests.fakes import FakeUpstream, Reply, chat_body
from evals.skill_bench.tests.method_contract import contract_cell, contract_items
from evals.skill_bench.tests.rollout_support import gateway_actor

PROMPTS = protocol.SKILL_BENCH_DIR / J.PROMPT_DIR
ORIGINALS = PROMPTS / "originals"
RB_PROMPTS = ORIGINALS / "reasoning-bank-ed80611" / "prompts.py.txt"
RB_EVALUATOR = ORIGINALS / "reasoning-bank-ed80611" / "evaluator.py.txt"
PAPER_SYSTEM = ORIGINALS / "paper_2509.25140_figure10_system.txt"
LUNA = protocol.MODELS["luna"]


def normal(text: str) -> str:
    """Whitespace-insensitive form (the PDF text layer flattens the figure's line breaks)."""
    return " ".join(text.split())


def rb_function_source(name: str) -> ast.FunctionDef:
    tree = ast.parse(RB_PROMPTS.read_text())
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def rb_string(function: str, variable: str) -> ast.expr:
    for node in ast.walk(rb_function_source(function)):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == variable
        ):
            return node.value
    raise AssertionError(f"{variable} not found in {function}")


def rb_system(function: str) -> str:
    value = rb_string(function, "system_msg")
    assert isinstance(value, ast.Constant) and isinstance(value.value, str)
    return value.value


# --- the system prompt: Figure 10 with one documented substitution ---------------------------


def test_system_prompt_is_figure_10_with_only_the_documented_substitution():
    paper = normal(PAPER_SYSTEM.read_text())
    ours = normal(J.system_prompt())
    substitution = J.judge_format()["substitution"]
    original, replacement = (
        normal(substitution["original"]),
        normal(substitution["replacement"]),
    )
    assert paper.count(original) == 1 and ours.count(replacement) == 1
    assert paper.replace(original, replacement) == ours
    # The replaced block is exactly the three web task types; the rest is verbatim.
    assert original.startswith("There are three types of tasks:")
    assert "3.Content modification" in original and "*IMPORTANT*" not in original
    assert replacement.startswith("There are two types of tasks:")
    assert "1. Answering or looking up information:" in replacement
    assert "2. Changing the state or taking an action:" in replacement
    for web in (
        "product",
        "map",
        "shop",
        "Site navigation",
        "webpage or configuration",
    ):
        assert web not in replacement


def test_the_strict_paragraph_of_the_later_repo_version_is_left_out():
    """Q22: the repo's build_text_eval_prompt adds '*Strictness rules*'; J follows the paper."""
    repo = rb_system("build_text_eval_prompt")
    assert "*Strictness rules*" in repo
    strict = repo[repo.index("*Strictness rules*") : repo.index("*IMPORTANT*")]
    without = normal(repo.replace(strict, ""))
    # The PDF text layer drops the space after two list numbers; otherwise identical.
    paper = re.sub(r"\b([13])\.(\S)", r"\1. \2", normal(PAPER_SYSTEM.read_text()))
    assert without == paper
    assert "Strictness" not in J.system_prompt()
    assert "A false success is more harmful" not in J.system_prompt()
    # The repo's vision variant has no strict paragraph either (and no stray quote).
    vision = normal(rb_system("build_vision_eval_prompt"))
    assert vision == without.replace('process>" Status', "process> Status")


def test_system_prompt_keeps_figure_10s_layout_from_the_repo_copy():
    repo = rb_system("build_text_eval_prompt")
    strict = repo[repo.index("*Strictness rules*") : repo.index("*IMPORTANT*")]
    repo_lines = repo.replace(strict, "").rstrip("\n").split("\n")
    ours = J.system_prompt().split("\n")
    # Same lines outside the rewritten task types (Figure 10's intro and output format).
    assert ours[:2] == repo_lines[:2]
    assert ours[-6:] == repo_lines[-6:]
    assert ours[-2:] == [
        'Thoughts: <your thoughts and reasoning process>"',
        'Status: "success" or "failure"',
    ]


def test_originals_match_their_recorded_hashes_and_licence():
    provenance = J.judge_format()["provenance"]
    for key in ("system_original", "input_original"):
        path = PROMPTS / provenance["paper"][key]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == provenance["paper"][f"{key}_sha256"]
    for rel, entry in provenance["repo"]["files"].items():
        data = (PROMPTS / rel).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        # The git blob id of the verbatim copy is the one at ed80611.
        blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        assert blob == entry["git_blob"]
    assert (
        "Apache License"
        in (ORIGINALS / "reasoning-bank-ed80611" / "LICENSE").read_text()
    )
    assert provenance["repo"]["commit"].startswith("ed80611")
    for rel in (
        "system.md",
        "user.md",
        "format.json",
        "originals/paper_2509.25140_figure10_system.txt",
    ):
        assert protocol.is_fingerprinted(Path(J.PROMPT_DIR) / rel)


# --- the input: RB's text-evaluation prompt ------------------------------------------------------


def test_user_template_is_the_repos_text_evaluation_prompt():
    value = rb_string("build_text_eval_prompt", "prompt")
    assert isinstance(value, ast.JoinedStr)
    names = {
        "intent": "{intent}",
        "last_actions": "{actions}",
        "cap": "{observations}",
    }
    rebuilt = []
    for part in value.values:
        if isinstance(part, ast.Constant):
            rebuilt.append(str(part.value))
        else:
            assert isinstance(part, ast.FormattedValue)
            expr = ast.unparse(part.value)
            rebuilt.append(names.get(expr, "{response}"))
            if expr not in names:
                assert expr == "response if response else 'N/A'"
    assert "".join(rebuilt) == J.user_template()
    assert J.judge_format()["no_response"] == "N/A"


def test_action_and_observation_formats_are_the_repo_evaluators():
    source = RB_EVALUATOR.read_text()
    assert 'f"{idx+1}: <think>{think}</think>\\n   <action>{act}</action>\\n"' in source
    assert 'f"{idx+1}: <action>{act}</action>\\n"' in source
    assert "captions[-5:]" in source and J.judge_format()["max_observations"] == 5
    assert '"\\n\\n---\\n\\n".join(' in source
    assert 'f"[Page state {i+1}/{len(last_caps)}]\\n{c}"' in source
    turns = [J.Turn("look first", ("go to shelf 1",)), J.Turn("", ("look",))]
    text, cut = J.render_actions(turns)
    assert cut == 0
    assert text == (
        "1: <think>look first</think>\n   <action>go to shelf 1</action>\n"
        "2: <action>look</action>\n"
    )


def view_of(item: Any) -> TrajectoryView:
    return item.trajectory()


def test_alfworld_input_numbers_actions_with_reasoning_and_keeps_the_last_five_observations():
    view = view_of(contract_items("alfworld")[0])
    built = J.build_judge_input(view, "alfworld")
    system, user = built.messages
    assert system == {"role": "system", "content": J.system_prompt()}
    text = user["content"]
    assert text.startswith("User Intent: put a mug in cabinet 1.\n\nAction History:\n")
    assert (
        "1: <think>The mug may be on the countertop.</think>\n   <action>go to countertop 1</action>\n"
        in text
    )
    # X7: turns without an action (a call without a reply; reasoning without an <action>
    # tag, for which ALFWorld played 'look') are left out before numbering, as RB's
    # evaluator skips empty-action steps, and counted.
    assert "<action></action>" not in text and "I would take the mug." not in text
    assert (
        "2: <think>…</think>\n   <action>take mug 1 from countertop 1</action>\n"
        in text
    )
    assert built.actions == 2 and built.skipped == 2 and built.observations == 5
    assert (
        "[Page state 1/5]" in text and "[Page state 5/5]\nYou pick up the mug 1" in text
    )
    assert text.endswith("Bot response to the user: N/A.")
    many = TrajectoryView(
        view.task_uid,
        view.instruction,
        view.first_messages,
        tuple(view.messages)
        + tuple({"role": "user", "content": f"obs {i}"} for i in range(9)),
    )
    text = J.build_judge_input(many, "alfworld").messages[1]["content"]
    assert "[Page state 1/5]\nobs 4" in text and "[Page state 5/5]\nobs 8" in text
    assert "obs 3" not in text


def test_long_observations_keep_head_and_tail_with_a_marked_cut():
    long = "HEAD" + "x" * 100_000 + "TAIL"
    view = TrajectoryView("t" + "0" * 20, "q", (), ({"role": "user", "content": long},))
    built = J.build_judge_input(view, "alfworld")
    text = built.messages[1]["content"]
    cap = J.judge_format()["observation_max_chars"]
    assert "HEAD" in text and "TAIL" in text
    assert re.search(r"\[\.\.\. [\d,]+ of 100,008 characters omitted \.\.\.\]", text)
    assert built.truncation["observations_chars_cut"] == len(long) - cap + len(
        J.judge_format()["truncation_marker"].format(
            omitted=built.truncation["observations_chars_cut"], total=len(long)
        )
    )
    kept, lost = J.cut_head_tail(long, cap)
    assert len(kept) == cap and lost > 0


def test_an_input_over_the_budget_is_cut_until_it_fits():
    limits = protocol.ModelLimits(LUNA, 40_000, 20_000)  # budget: 32,000 - 4,000 tokens
    words = " ".join(f"w{i}" for i in range(30_000))
    view = TrajectoryView(
        "t" + "0" * 20,
        "q",
        (),
        tuple(
            m
            for i in range(5)
            for m in (
                {
                    "role": "assistant",
                    "content": f"<think>{words}</think><action>a{i}</action>",
                },
                {"role": "user", "content": words},
            )
        ),
    )
    built, estimate = J.fit_judge_input(view, "alfworld", limits)
    assert estimate <= J.input_budget(limits) == 28_000
    assert built.truncation["cap"] is not None
    assert built.truncation["actions_chars_cut"] > 0
    small, small_estimate = J.fit_judge_input(
        view_of(contract_items("alfworld")[0]), "alfworld", limits
    )
    assert small.truncation["cap"] is None and small_estimate < 28_000


# --- parsing ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "status", "thoughts"),
    [
        ("Thoughts: it worked\nStatus: success", "success", "it worked"),
        (
            'Thoughts: the answer is wrong"\nStatus: "failure"',
            "failure",
            "the answer is wrong",
        ),
        (
            "Thoughts: a\nmore reasoning\nStatus: Success.",
            "success",
            "a\nmore reasoning",
        ),
        ("**Thoughts:** ok\n**Status:** success", "success", "ok"),
        ("**Status**: FAILURE", "failure", ""),
        ("  status : success", "success", ""),
        (
            "Thoughts: x\nStatus: failure\nStatus: success",
            "failure",
            "x",
        ),  # first decides
        ("Thoughts: x\nStatus: success or failure", None, "x"),
        ("Thoughts: x\nStatus: partially successful", None, "x"),
        ("The Status: success", None, ""),  # the label must start the line (RB)
        ("I think it succeeded.", None, ""),
        ("", None, ""),
        (None, None, ""),
    ],
)
def test_parse_judgement(reply, status, thoughts):
    judgement = J.parse_judgement(reply)
    assert judgement.status == status and judgement.parsed == (status is not None)
    assert judgement.thoughts == thoughts


# --- calls through the gateway ----------------------------------------------------------------


def cell_for(tmp_path: Path, benchmark: str = "alfworld", **kw: Any) -> CellContext:
    return contract_cell(
        tmp_path / "work",
        benchmark=benchmark,
        method="reasoningbank",
        gateway_base_url="http://gateway",
        run_id="cell-test-a1",
        **kw,
    )


async def judge_once(
    tmp_path: Path,
    replies: list[Reply],
    view: TrajectoryView,
    *,
    benchmark: str = "alfworld",
    **gateway: Any,
) -> tuple[J.Judge, J.JudgeVerdict | BaseException, FakeUpstream]:
    fake = FakeUpstream()
    fake.queue("chat/completions", *replies)
    async with gateway_actor(tmp_path, fake, **gateway) as (gw, _actor):
        judge = J.Judge(
            cell_for(tmp_path, benchmark), transport=httpx.ASGITransport(app=gw.app)
        )
        async with judge:
            try:
                return judge, await judge.judge(view), fake
            except Exception as exc:  # returned for the test to inspect
                return judge, exc, fake


def ledger(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "runs" / "cell-test-a1" / "gateway.jsonl"
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if '"event": "call"' in line
    ]


@pytest.mark.asyncio
async def test_a_verdict_goes_through_the_judge_role_with_the_cell_model(tmp_path):
    view = view_of(contract_items("alfworld")[0])
    judge, verdict, fake = await judge_once(
        tmp_path,
        [Reply(json=chat_body(content="Thoughts: done\nStatus: success"))],
        view,
    )
    assert isinstance(verdict, J.JudgeVerdict)
    assert verdict.success and verdict.status == "success" and verdict.parsed
    assert verdict.thoughts == "done" and verdict.failure is None
    assert verdict.task_uid == view.task_uid and verdict.actions == 2
    assert verdict.skipped_turns == 2
    body = fake.calls("chat/completions")[0].json
    assert body["model"] == LUNA and "temperature" not in body
    assert body["messages"] == J.build_judge_input(view, "alfworld").messages
    row = ledger(tmp_path)[0]
    assert (row["role"], row["item"], row["phase"], row["model"]) == (
        "judge",
        view.task_uid,
        "learning",
        LUNA,
    )
    assert verdict.gateway_call_id == row["call_id"]
    assert judge.counters.to_dict() == {
        "calls": 1,
        "success": 1,
        "failure": 0,
        "unparseable": 0,
        "model_failures": 0,
        "truncated_inputs": 0,
    }
    records = [
        json.loads(line)
        for line in (tmp_path / "work" / "judge.jsonl").read_text().splitlines()
    ]
    assert [(r["task_uid"], r["status"], r["parsed"]) for r in records] == [
        (view.task_uid, "success", True)
    ]


@pytest.mark.asyncio
async def test_an_unparseable_verdict_is_a_counted_failure(tmp_path):
    view = view_of(contract_items("alfworld")[0])
    judge, verdict, _ = await judge_once(
        tmp_path, [Reply(json=chat_body(content="I believe the agent did well."))], view
    )
    assert isinstance(verdict, J.JudgeVerdict)
    assert not verdict.success and verdict.status == "failure" and not verdict.parsed
    assert verdict.raw == "I believe the agent did well."
    assert judge.counters.unparseable == 1 and judge.counters.failure == 1


@pytest.mark.asyncio
async def test_a_truncated_reply_is_parsed_like_any_reply_and_counted(tmp_path):
    view = view_of(contract_items("alfworld")[0])
    reply = Reply(
        json=chat_body(content="Thoughts: ok\nStatus: success", finish_reason="length")
    )
    judge, verdict, _ = await judge_once(tmp_path, [reply], view)
    assert isinstance(verdict, J.JudgeVerdict)
    assert verdict.success and verdict.failure is not None
    assert verdict.failure["kind"] == "model" and judge.counters.model_failures == 1


@pytest.mark.asyncio
async def test_a_call_without_a_reply_by_the_models_fault_is_a_counted_failure(
    tmp_path,
):
    view = view_of(contract_items("alfworld")[0])
    judge, verdict, _ = await judge_once(
        tmp_path, [Reply(delay_s=3.0)], view, deadline_s=1.0
    )
    assert isinstance(verdict, J.JudgeVerdict)
    assert verdict.status == "failure" and not verdict.parsed and verdict.raw is None
    assert verdict.failure is not None and verdict.failure["code"] == "model_timeout"
    assert verdict.http_status == 504
    assert judge.counters.to_dict()["unparseable"] == judge.counters.model_failures == 1


@pytest.mark.asyncio
async def test_an_infrastructure_failure_raises(tmp_path):
    view = view_of(contract_items("alfworld")[0])
    _, outcome, _ = await judge_once(tmp_path, [Reply(503)], view, max_retries=0)
    assert isinstance(outcome, ActorCallError)
    assert outcome.failure.kind is FailureKind.INFRA and outcome.failure.role == "judge"
    assert not (tmp_path / "work" / "judge.jsonl").exists()


@pytest.mark.asyncio
async def test_the_judge_accepts_only_the_method_visible_view(tmp_path):
    from evals.skill_bench.tests.test_agent_trajectory import sample_trajectory

    judge = J.Judge(cell_for(tmp_path), record_path=None)
    private = sample_trajectory()  # holds the scorer's result and private step data
    with pytest.raises(TypeError, match="TrajectoryView"):
        await judge.judge(private)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        await judge.judge(private.task_view())  # type: ignore[arg-type]
    await judge.aclose()
    params = list(__import__("inspect").signature(J.Judge.judge).parameters)
    assert params == ["self", "trajectory"]  # no way to hand it a label


def test_task_view_and_verdict_types_are_what_s1_methods_use():
    view = TaskView("t" + "0" * 20, "q", ())
    assert J.JUDGE_ROLE == "judge" and J.STATUSES == ("success", "failure")
    assert isinstance(view, TaskView)
    verdict_fields = set(J.JudgeVerdict.__dataclass_fields__)
    assert {
        "success",
        "status",
        "parsed",
        "thoughts",
        "failure",
        "gateway_call_id",
    } <= verdict_fields
    assert not verdict_fields & {"score", "won", "gold", "answer", "answers"}


# --- learning checkpoints (contract item 6) --------------------------------------------------


@pytest.mark.asyncio
async def test_the_judges_counts_resume_from_a_checkpoint(tmp_path):
    """A method restoring a learning checkpoint puts J's counters back in place (a
    method sharing ``judge.counters`` sees them) and J's next call has the index an
    uninterrupted run's would."""
    view = view_of(contract_items("alfworld")[0])
    first, verdict, _ = await judge_once(
        tmp_path / "a",
        [Reply(json=chat_body(content="Thoughts: ok\nStatus: success"))],
        view,
    )
    assert isinstance(verdict, J.JudgeVerdict)
    recorded = first.counters.to_dict()

    fake = FakeUpstream()
    fake.queue(
        "chat/completions",
        Reply(json=chat_body(content="Thoughts: no\nStatus: failure")),
    )
    async with gateway_actor(tmp_path / "b", fake) as (gw, _actor):
        judge = J.Judge(
            cell_for(tmp_path / "b"), transport=httpx.ASGITransport(app=gw.app)
        )
        shared = judge.counters
        judge.restore(recorded)
        assert shared is judge.counters and shared.to_dict() == recorded
        async with judge:
            await judge.judge(view)
    assert judge.counters.to_dict() == {**recorded, "calls": 2, "failure": 1}
    assert judge._calls == 2
    with pytest.raises(TypeError):
        judge.restore({**recorded, "unknown": 1})
    assert J.JudgeCounters.from_dict(recorded) == first.counters
