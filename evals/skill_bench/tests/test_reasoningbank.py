"""ReasoningBank: prompt faithfulness, k = 1 retrieval, the frozen test phase, failure
counting and label isolation, offline through the real gateway."""

from __future__ import annotations

from evals.skill_bench.methods.base import FrozenMemoryMismatch

import ast
import copy
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

from evals.skill_bench import protocol
from evals.skill_bench.agent.actor import ActorCallError
from evals.skill_bench.infra.failures import FailureKind
from evals.skill_bench.methods import awm as A
from evals.skill_bench.methods import judge as J
from evals.skill_bench.methods import reasoningbank as R
from evals.skill_bench.methods.base import CellContext, Injection, TaskView
from evals.skill_bench.tests.fakes import FakeUpstream, Reply, running_gateway
from evals.skill_bench.tests.method_contract import (
    ContractItem,
    contract_cell,
    contract_items,
    exercise_method,
)
from evals.skill_bench.tests.rollout_support import text_reply
from evals.skill_bench.tests.s1_support import (
    Router,
    fstring_template,
    function,
    items_reply,
    judged,
    ledger,
    module_ast,
    normal,
    originals_match,
    read_jsonl,
    string_constant,
)

PROMPTS = protocol.SKILL_BENCH_DIR / R.PROMPT_DIR
ORIGINALS = PROMPTS / "originals"
REPO = ORIGINALS / "reasoning-bank-ed80611"
NOTICE = PROMPTS.parent / "NOTICE"
RUN = "rb-test-a1"
PANEL_FILE = {
    "success": "paper_2509.25140_figure9_success_system.txt",
    "failure": "paper_2509.25140_figure9_failure_system.txt",
}
REPO_PROMPT = {"success": "SUCCESSFUL_SI", "failure": "FAILED_SI"}


def strip_emphasis(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*|(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1\2", text)


def paper_text(name: str) -> str:
    """A text-layer original with the layer's artefacts undone (format.json text_layer_note)."""
    text = (ORIGINALS / name).read_text(encoding="utf-8")
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("’", "'")
    text = re.sub(r"(^|\s)-(?=\S)", r"\1- ", text)
    return normal(text)


# --- the extraction instructions: Figure 9 with the two documented substitutions ----------------


@pytest.mark.parametrize("status", ["success", "failure"])
def test_system_prompts_are_figure_9_with_only_the_documented_substitutions(status):
    paper = paper_text(PANEL_FILE[status])
    for sub in R.rb_format()["system_prompts"]["substitutions"]:
        assert paper.count(sub["original"]) == 1, sub["original"]
        paper = paper.replace(sub["original"], sub["replacement"])
    ours = R.system_prompt(status)
    assert normal(strip_emphasis(ours)) == paper
    assert "web" not in ours.lower() and "website" not in ours.lower()
    assert ours.startswith("You are an expert AI agent analyst.")
    assert (
        "Do not mention specific entities, file names, queries, or string contents"
        in ours
    )


def figure_emphasis(panel: str) -> tuple[set[str], set[str]]:
    """Bold and italic runs of one panel's system instruction (from the span dump)."""
    runs: list[tuple[str, str]] = []
    for line in (
        (ORIGINALS / "paper_2509.25140_figure9_spans.txt").read_text().splitlines()
    ):
        if not line.startswith(panel + " "):
            continue
        spans = re.findall(r"\[([^\]]+)\]([^\[]*)", line)
        if spans[0][1].strip() in ("System Instruction", "Input Prompt") or spans[0][
            1
        ].startswith(("Query:", "Trajectory:")):
            continue
        for font, text in spans:
            style = (
                "bold"
                if font.endswith("Heavy")
                else "italic"
                if "Oblique" in font
                else "plain"
            )
            runs.append((style, text))
        if not runs[-1][1].endswith(" "):
            runs.append(("plain", " "))
    merged: list[list[str]] = []
    for style, text in runs:
        if merged and merged[-1][0] == style:
            merged[-1][1] += text
        else:
            merged.append([style, text])
    bold = {normal(t) for s, t in merged if s == "bold"}
    italic = {normal(t) for s, t in merged if s == "italic"}
    return bold, italic


@pytest.mark.parametrize("status", ["success", "failure"])
def test_emphasis_is_the_figures_and_includes_the_repos(status):
    ours = R.system_prompt(status)
    bold = set(re.findall(r"\*\*(.+?)\*\*", ours))
    italic = set(re.findall(r"(?<!\*)\*([^*\n]+)\*(?!\*)", ours))
    assert (bold, italic) == figure_emphasis(status)
    repo = string_constant(
        module_ast(REPO / "memory_instruction.py.txt"), REPO_PROMPT[status]
    )
    assert set(re.findall(r"\*\*(.+?)\*\*", repo)) < bold
    assert set(re.findall(r"(?<!\*)\*([^*\n]+)\*(?!\*)", repo)) == italic


def shape(line: str) -> str:
    match = re.match(r"^(#+ |  - |```|# Memory Item i$)", line)
    return match.group(1) if match else ("text" if line else "empty")


@pytest.mark.parametrize("status", ["success", "failure"])
def test_layout_is_the_repo_copys_and_the_wording_the_papers(status):
    repo = string_constant(
        module_ast(REPO / "memory_instruction.py.txt"), REPO_PROMPT[status]
    )
    repo_lines = repo.strip("\n").split("\n")
    ours = R.system_prompt(status).split("\n")
    assert [shape(line) for line in ours] == [
        shape(line.rstrip()) for line in repo_lines
    ]
    same = [a for a, b in zip(ours, repo_lines) if a == b.rstrip()]
    assert "## Guidelines" in same and "## Output Format" in same and "```" in same
    # Where the repo's later copy words it differently, the paper wins.
    assert "Prefer concrete, actionable" in repo and "Prefer concrete" not in "\n".join(
        ours
    )
    assert "when NOT to use" in repo and "when NOT to use" not in "\n".join(ours)
    assert "## Description <one sentence summary of the memory item>" in ours
    # The repo's SWE-bench copy keeps the paper's wording of the notes in the same layout.
    swe = string_constant(
        module_ast(REPO / "swebench_instruction.py.txt"), REPO_PROMPT[status]
    )
    assert (
        "  - Do not mention specific websites, queries, or string contents, but rather "
        "focus on the generalizable insights." in swe
    )


def test_user_template_step_format_and_joiners_are_induce_memorys():
    tree = module_ast(REPO / "induce_memory.py.txt")
    main = function(tree, "main")
    templates = [
        node.value
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "trajectory"
        and isinstance(node.value, ast.JoinedStr)
    ]
    assert len(templates) == 1
    names = {"ex['query']": "{query}", "trajectory": "{trajectory}"}
    assert fstring_template(templates[0], names) == R.rb_format()["user_template"]
    step = next(
        node
        for node in ast.walk(function(tree, "format_trajectory"))
        if isinstance(node, ast.JoinedStr)
    )
    fmt = R.rb_format()
    assert fstring_template(step, {"t": "{think}", "acts": "{action}"}) == fmt["step"]
    source = (REPO / "induce_memory.py.txt").read_text()
    assert "return '\\n\\n'.join(trajectory)" in source and fmt["step_joiner"] == "\n\n"
    # The repo appends J's reasoning to the extraction input; the paper does not, nor do we.
    assert (
        'trajectory += f"\\n\\nThe task {status_label} because: {autoeval_thoughts}"'
        in source
    )
    assert "because" not in fmt["user_template"]
    # AWM's format_trajectory, which RB copied, has the same step format.
    assert A.awm_format()["step"] == fmt["step"]


def test_retrieval_wrapper_is_memory_managements_with_one_substitution():
    tree = module_ast(REPO / "memory_management.py.txt")
    wrapper = next(
        node
        for node in ast.walk(function(tree, "get_detailed_instruct"))
        if isinstance(node, ast.JoinedStr)
    )
    fmt = R.rb_format()
    assert (
        fstring_template(wrapper, {"task_description": "{task}", "query": "{query}"})
        == fmt["retrieval_query"]
    )
    task = next(
        node.value.value
        for node in ast.walk(function(tree, "screening"))
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "task"
        and isinstance(node.value, ast.Constant)
    )
    assert isinstance(task, str)
    assert task.count("web navigation") == 1
    assert task.replace("web navigation", "task") == fmt["retrieval_task"]
    assert R.retrieval_query("heat an egg.") == (
        "Instruct: Given the prior task queries, your task is to analyze a current "
        "query's intent and select relevant prior queries that could help resolve it.\n"
        "Query: heat an egg."
    )
    run = (REPO / "run.py.txt").read_text()
    assert "select_memory(n=1," in run and fmt["top_k"] == 1


def test_injection_is_the_papers_sentence_then_title_and_content():
    preamble = R.rb_format()["injection_preamble"]
    agent = (REPO / "agent.py.txt").read_text()
    assert f'"{preamble}"' in agent
    assert preamble in paper_text(
        "paper_2509.25140_appendix_a2_injection_consolidation.txt"
    )
    assert "each item represented by its title and content" in paper_text(
        "paper_2509.25140_appendix_a2_extraction_retrieval.txt"
    )
    items = [R.MemoryItem("T1", "D1", "C1"), R.MemoryItem("T2", "D2", "C2\nmore")]
    assert R.render_injection(items) == (
        f"{preamble}\n\n# Memory Item 1\n## Title T1\n## Content C1\n\n"
        "# Memory Item 2\n## Title T2\n## Content C2\nmore"
    )
    assert R.render_injection([]) == ""


def test_originals_hashes_blobs_licence_and_fingerprint():
    provenance = R.rb_format()["provenance"]
    checked = originals_match(PROMPTS, provenance["paper"]["originals"])
    checked += originals_match(PROMPTS, provenance["repo"]["files"])
    assert len(checked) == 12
    assert provenance["repo"]["commit"].startswith("ed80611")
    assert "Apache License" in (REPO / "LICENSE").read_text()
    assert "reasoningbank/originals/reasoning-bank-ed80611/" in NOTICE.read_text()
    for rel in ("success_system.md", "failure_system.md", "format.json"):
        assert protocol.is_fingerprinted(Path(R.PROMPT_DIR) / rel)


def code_strings(path: Path) -> str:
    """Every string literal of a module except its docstrings."""
    tree = ast.parse(path.read_text())
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
    return "\n".join(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    )


def test_no_model_visible_literal_in_the_code():
    code = code_strings(protocol.SKILL_BENCH_DIR / "methods" / "reasoningbank.py")
    code += code_strings(protocol.SKILL_BENCH_DIR / "methods" / "awm.py")
    for literal in (
        "Below are some memory items",
        "Instruct:",
        "**Query:**",
        "# Memory Item {i}",
        "<observation>",
        "## Concrete Examples",
        "Summary Workflows",
        "expert AI agent analyst",
        "Actions:",
        "<think>",
        "Workflow 1",
    ):
        assert literal not in code, literal


# --- parsing the extraction reply ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "titles", "incomplete"),
    [
        (items_reply("A", "B"), ["A", "B"], 0),
        (
            "# Memory Item 1\n## Title: A\n## Description: d\n## Content: c",
            ["A"],
            0,
        ),
        ("**Title:** A\n**Description:** d\n**Content:** c1\nc2", ["A"], 0),
        ("## **Title** A\n## Content\nline one\nline two", ["A"], 0),
        ("# Memory Item 1\n## Title A\n## Description d", [], 1),  # no content
        ("## Title A\n## Content c\n## Title B\n## Content d", ["A", "B"], 0),
        ("1. **Title**: A\n   **Description**: d\n   **Content**: c", ["A"], 0),
        ("- Title: A\n- Content: c\n- Title: B\n- Content: d", ["A", "B"], 0),
        (
            "Why it worked: good.\n\n**Memory Item 1**\n**Title:** A\n**Content:** c\n\n"
            "**Memory Item 2**\n**Title:** B\n**Content:** d",
            ["A", "B"],
            0,
        ),
        ("## Title\n\nX\n\n## Content\n\nY", ["X"], 0),
        ("I could not find anything useful.", [], 0),
        ("", [], 0),
        (None, [], 0),
    ],
)
def test_parse_memory_items(reply, titles, incomplete):
    parsed = R.parse_memory_items(reply)
    assert [i.title for i in parsed.items] == titles
    assert parsed.incomplete == incomplete


def test_a_bold_item_header_ends_the_previous_content():
    parsed = R.parse_memory_items(
        "**Memory Item 1**\n**Title:** A\n**Content:** c\n\n**Memory Item 2**\n"
        "**Title:** B\n**Content:** d\nMemory item 3 is not a header here."
    )
    assert [i.content for i in parsed.items] == [
        "c",
        "d\nMemory item 3 is not a header here.",
    ]


def test_parsed_items_keep_their_fields():
    parsed = R.parse_memory_items(
        "```\n# Memory Item 1\n## Title Check the receptacle first\n"
        "## Description when an object is hidden\n## Content Open it,\nthen look.\n```"
    )
    assert parsed.items == (
        R.MemoryItem(
            "Check the receptacle first",
            "when an object is hidden",
            "Open it,\nthen look.",
        ),
    )


# --- the extraction input --------------------------------------------------------------------------


def test_alfworld_input_has_steps_observations_and_skips_turns_without_an_action():
    item = contract_items("alfworld")[0]
    steps = A.trajectory_steps(item.trajectory())
    assert (
        steps.skipped == 2
    )  # the empty turn of a call without a reply, the untagged turn
    messages, cut = R.extraction_messages(item.view.instruction, steps, "success")
    assert cut == 0
    system, user = messages
    assert system == {"role": "system", "content": R.system_prompt("success")}
    text = user["content"]
    assert text.startswith(
        "**Query:** put a mug in cabinet 1.\n\n**Trajectory:**\n<observation>\n"
        "-= Welcome to TextWorld, ALFRED! =-"
    )
    assert (
        "<think>\nThe mug may be on the countertop.\n</think>\n<action>\ngo to countertop 1\n"
        "</action>\n<observation>\nYou arrive at countertop 1." in text
    )
    assert text.endswith(
        "<action>\ntake mug 1 from countertop 1\n</action>\n<observation>\n"
        "You pick up the mug 1 from the countertop 1.\n</observation>"
    )
    assert (
        "I would take the mug." not in text
    )  # the untagged turn is skipped, as RB does
    failure = R.extraction_messages(item.view.instruction, steps, "failure")[0]
    assert failure[0]["content"] == R.system_prompt("failure")


def test_long_observations_keep_head_and_tail():
    long = "HEAD" + "x" * 100_000 + "TAIL"
    view = ContractItem(
        TaskView("t" + "1" * 20, "q", ({"role": "user", "content": "start"},)),
        (
            {"role": "user", "content": "start"},
            {"role": "assistant", "content": "<action>look</action>"},
            {"role": "user", "content": long},
        ),
    ).trajectory()
    text = R.extraction_messages("q", A.trajectory_steps(view), "success")[0][1][
        "content"
    ]
    assert "HEAD" in text and "TAIL" in text
    assert re.search(r"\[\.\.\. [\d,]+ of 100,008 characters omitted \.\.\.\]", text)
    assert len(text) < 41_000


def test_an_input_over_the_budget_is_cut_until_it_fits():
    limits = protocol.ModelLimits(protocol.MODELS["luna"], 40_000, 20_000)
    words = " ".join(f"w{i}" for i in range(30_000))
    messages: list[dict[str, Any]] = [{"role": "user", "content": "start"}]
    for i in range(5):
        messages += [
            {
                "role": "assistant",
                "content": f"<think>{words}</think><action>a{i}</action>",
            },
            {"role": "user", "content": words},
        ]
    view = ContractItem(
        TaskView("t" + "2" * 20, "q", (messages[0],)), tuple(messages)
    ).trajectory()
    built, estimate, cap, cut = R.fit_extraction(
        "q", A.trajectory_steps(view), "success", limits
    )
    assert estimate <= J.input_budget(limits) and cap is not None and cut > 0
    small = R.fit_extraction(
        "q",
        A.trajectory_steps(contract_items("alfworld")[0].trajectory()),
        "success",
        limits,
    )
    assert small[2] is None


# --- the method through the gateway -------------------------------------------------------------------


def alf_item(uid: str, instruction: str, action: str = "look") -> ContractItem:
    first = {
        "role": "user",
        "content": f"-= Welcome =-\n\nYour task is to: {instruction}\n\nAdmissible actions:\n'look'",
    }
    messages = (
        first,
        {
            "role": "assistant",
            "content": f"<think>try {action}</think><action>{action}</action>",
        },
        {"role": "user", "content": f"You did {action}."},
    )
    return ContractItem(TaskView(uid, instruction, (first,)), messages)


Q1, Q2, Q3, Q4 = (
    "put a mug in cabinet 1.",
    "heat an egg and put it in the fridge.",
    "cool an egg and put it on the table.",
    "look at a book under the lamp.",
)
VECTORS = {
    Q1: [1.0, 0.0, 0.0],
    Q2: [0.0, 1.0, 0.0],
    R.retrieval_query(Q1): [0.9, 0.1, 0.0],
    R.retrieval_query(Q2): [0.2, 0.8, 0.0],
    R.retrieval_query(Q3): [0.1, 0.9, 0.0],  # nearest: Q2
    R.retrieval_query(Q4): [2.0, 2.0, 0.0],  # a tie between Q1 and Q2
}


def rb_cell(tmp_path: Path, benchmark: str = "alfworld", **kw: Any) -> CellContext:
    return contract_cell(
        tmp_path / "work",
        benchmark=benchmark,
        method="reasoningbank",
        gateway_base_url="http://gateway",
        run_id=RUN,
        **kw,
    )


async def learn(
    method: R.ReasoningBank, items: Sequence[ContractItem]
) -> list[Injection]:
    injections = []
    for item in items:
        injections.append(await method.before_task(copy.deepcopy(item.view)))
        for message in item.messages:
            await method.on_message(item.view.task_uid, copy.deepcopy(message))
        await method.after_task(item.trajectory())
        await method.wait_until_idle(5.0)
    return injections


@pytest.mark.asyncio
async def test_learning_extracts_from_successes_and_failures_and_retrieves_k1(tmp_path):
    router = Router(
        judge=[
            judged("success", "THOUGHTS-SENTINEL-1"),
            judged("failure", "THOUGHTS-SENTINEL-2"),
        ],
        learner=[items_reply("A1", "A2", "A3", "A4"), items_reply("B1")],
        vectors=VECTORS,
    )
    fake = FakeUpstream(responder=router)
    async with running_gateway(tmp_path, fake) as (gw, _client):
        method = R.ReasoningBank(transport=httpx.ASGITransport(app=gw.app))
        await method.setup(rb_cell(tmp_path))
        try:
            first, second = await learn(
                method, [alf_item("t" + "a" * 20, Q1), alf_item("t" + "b" * 20, Q2)]
            )
            assert first == Injection()  # an empty bank: no retrieval call
            assert not second.empty  # the bank had Q1's items by then
            third = await method.before_task(TaskView("t" + "c" * 20, Q3, ()))
            tie = await method.before_task(TaskView("t" + "d" * 20, Q4, ()))
            snapshot = await method.snapshot()
        finally:
            await method.close()
    # J judged both, through the judge role, with the cell's model.
    assert len(router.calls["judge"]) == 2
    success_call, failure_call = router.calls["learner"]
    assert success_call["messages"][0]["content"] == R.system_prompt("success")
    assert failure_call["messages"][0]["content"] == R.system_prompt("failure")
    user = success_call["messages"][1]["content"]
    assert user.startswith(f"**Query:** {Q1}\n\n**Trajectory:**\n<observation>\n")
    assert (
        "<action>\nlook\n</action>\n<observation>\nYou did look.\n</observation>"
        in user
    )
    for call in router.calls["learner"]:  # J's reasoning is not appended (H7.5)
        wire = json.dumps(call)
        assert "THOUGHTS-SENTINEL" not in wire and "because" not in wire
        assert (
            "temperature" not in call and "max_tokens" in call
        )  # the gateway's F9 cap
    # The stored vectors are those of the unwrapped queries; retrieval embeds the wrapped one.
    assert router.embedded() == [
        [Q1],
        [R.retrieval_query(Q2)],
        [Q2],
        [R.retrieval_query(Q3)],
        [R.retrieval_query(Q4)],
    ]
    # {query, status, memory_items}; at most 3 items per task, append only.
    entries = snapshot["entries"]
    assert [(e["query"], e["status"]) for e in entries] == [
        (Q1, "success"),
        (Q2, "failure"),
    ]
    assert [i["title"] for i in entries[0]["memory_items"]] == ["A1", "A2", "A3"]
    assert [i["title"] for i in entries[1]["memory_items"]] == ["B1"]
    assert set(entries[0]["memory_items"][0]) == {
        "id",
        "title",
        "description",
        "content",
    }
    counters = snapshot["counters"]
    assert counters["items_dropped_over_limit"] == 1 and counters["entries"] == 2
    assert counters["judged_success"] == counters["judged_failure"] == 1
    assert snapshot["judge"]["calls"] == 2
    # k = 1: the most similar prior task, all of its items, title and content only.
    assert third.item_ids == ("t" + "b" * 20 + "#1",)
    assert third.text == R.render_injection([R.MemoryItem("B1", "", "content of B1.")])
    assert "DESC-" not in third.text  # no description
    assert tie.item_ids == tuple(
        f"t{'a' * 20}#{i}" for i in (1, 2, 3)
    )  # tie: the earlier
    # Roles and phases in the gateway ledger.
    rows = ledger(tmp_path, RUN)
    assert sorted({r["role"] for r in rows}) == ["embedding", "judge", "learner"]
    assert {r["phase"] for r in rows} == {"learning"}
    assert all(
        r["model"] in (protocol.MODELS["luna"], protocol.EMBEDDING_MODEL) for r in rows
    )
    extractions = read_jsonl(tmp_path / "work" / R.EXTRACTIONS_FILE)
    assert [(x["status"], x["items_kept"], x["stored"]) for x in extractions] == [
        ("success", 3, True),
        ("failure", 1, True),
    ]


@pytest.mark.asyncio
async def test_the_test_phase_is_read_only_and_a_restart_resumes_the_frozen_bank(
    tmp_path,
):
    router = Router(
        judge=[judged("success"), judged("success")],
        learner=[items_reply("A1"), items_reply("B1", "B2")],
        vectors=VECTORS,
    )
    fake = FakeUpstream(responder=router)
    async with running_gateway(tmp_path, fake) as (gw, _client):
        transport = httpx.ASGITransport(app=gw.app)
        method = R.ReasoningBank(transport=transport)
        await method.setup(rb_cell(tmp_path))
        try:
            await learn(
                method, [alf_item("t" + "a" * 20, Q1), alf_item("t" + "b" * 20, Q2)]
            )
            await method.finalize()
            await method.wait_until_idle(5.0)
            frozen = await method.snapshot()
            assert frozen["frozen"] is True
            injection = await method.before_task(TaskView("t" + "c" * 20, Q3, ()))
            with pytest.raises(RuntimeError, match="read-only"):
                await method.after_task(alf_item("t" + "e" * 20, Q3).trajectory())
            assert await method.snapshot() == frozen
        finally:
            await method.close()
        assert injection.item_ids == (f"t{'b' * 20}#1", f"t{'b' * 20}#2")
        # A harness restarted in the test phase resumes the same bank.
        resumed = R.ReasoningBank(transport=transport)
        await resumed.setup(rb_cell(tmp_path, phase="test"))
        try:
            assert await resumed.snapshot() == frozen
            again = await resumed.before_task(TaskView("t" + "c" * 20, Q3, ()))
        finally:
            await resumed.close()
        assert again == injection
    test_rows = [r for r in ledger(tmp_path, RUN) if r["phase"] == "test"]
    assert [r["role"] for r in test_rows] == ["embedding", "embedding"]
    assert len(router.calls["learner"]) == 2  # no learning in the test phase
    # The frozen bank is what resume checks against.
    bank = tmp_path / "work" / R.BANK_FILE
    bank.write_text(bank.read_text().replace("B2", "B3"))
    with pytest.raises(FrozenMemoryMismatch, match="does not match"):
        await R.ReasoningBank().setup(rb_cell(tmp_path, phase="test"))


@pytest.mark.asyncio
async def test_setup_never_starts_on_stale_or_missing_memory(tmp_path):
    with pytest.raises(FrozenMemoryMismatch, match="no frozen memory"):
        await R.ReasoningBank().setup(rb_cell(tmp_path, phase="test"))
    (tmp_path / "work").mkdir(exist_ok=True)
    (tmp_path / "work" / R.BANK_FILE).write_text("{}\n")
    with pytest.raises(RuntimeError, match="empty work_dir"):
        await R.ReasoningBank().setup(rb_cell(tmp_path))


@pytest.mark.asyncio
async def test_an_empty_bank_injects_nothing_and_makes_no_call(tmp_path):
    router = Router()
    fake = FakeUpstream(responder=router)
    async with running_gateway(tmp_path, fake) as (gw, _client):
        method = R.ReasoningBank(transport=httpx.ASGITransport(app=gw.app))
        await method.setup(rb_cell(tmp_path))
        try:
            assert (
                await method.before_task(TaskView("t" + "a" * 20, Q1, ()))
                == Injection()
            )
        finally:
            await method.close()
    assert fake.requests == []


@pytest.mark.asyncio
async def test_model_failures_are_counted_and_survived(tmp_path):
    router = Router(
        judge=[
            judged("success"),
            judged("failure"),
            "no verdict here",
            judged("success"),
        ],
        learner=[
            Reply(delay_s=3.0),  # no reply within the deadline: a MODEL timeout
            text_reply(
                items_reply("T1"), finish_reason="length"
            ),  # truncated but parsed
            "I have no memory items to offer.",  # unparseable
            text_reply("", finish_reason="content_filter"),
        ],
        vectors=VECTORS,
    )
    fake = FakeUpstream(responder=router)
    async with running_gateway(tmp_path, fake, deadline_s=1.0) as (gw, _client):
        method = R.ReasoningBank(transport=httpx.ASGITransport(app=gw.app))
        await method.setup(rb_cell(tmp_path))
        try:
            await learn(
                method,
                [alf_item(f"t{c * 20}", Q1 if c in "ab" else Q2) for c in "abcd"],
            )
            snapshot = await method.snapshot()
        finally:
            await method.close()
    c = snapshot["counters"]
    assert c["extractions"] == 4 and c["extraction_no_reply"] == 1
    assert c["extraction_model_failures"] == 3  # timeout, length, content filter
    assert (
        c["unparseable_extractions"] == 2
    )  # the prose reply and the empty filtered one
    assert c["entries"] == 1 and [
        e["memory_items"][0]["title"] for e in snapshot["entries"]
    ] == ["T1"]
    assert snapshot["judge"]["unparseable"] == 1
    rows = read_jsonl(tmp_path / "work" / R.EXTRACTIONS_FILE)
    assert [r["failure"]["code"] if r["failure"] else None for r in rows] == [
        "model_timeout",
        "truncated",
        None,
        "content_filter",
    ]


@pytest.mark.asyncio
async def test_infrastructure_failures_raise(tmp_path):
    fake = FakeUpstream(
        responder=Router(judge=[judged("success")], learner=[Reply(503)])
    )
    async with running_gateway(tmp_path, fake, max_retries=0) as (gw, _client):
        method = R.ReasoningBank(transport=httpx.ASGITransport(app=gw.app))
        await method.setup(rb_cell(tmp_path))
        try:
            with pytest.raises(ActorCallError) as info:
                await learn(method, [alf_item("t" + "a" * 20, Q1)])
        finally:
            await method.close()
    assert info.value.failure.kind is FailureKind.INFRA
    assert info.value.failure.role == "learner"

    fake = FakeUpstream(
        responder=Router(judge=[judged("success")], learner=[items_reply("A")])
    )
    fake.queue("embeddings", Reply(503))
    async with running_gateway(tmp_path / "e", fake, max_retries=0) as (gw, _client):
        method = R.ReasoningBank(transport=httpx.ASGITransport(app=gw.app))
        await method.setup(rb_cell(tmp_path / "e"))
        try:
            with pytest.raises(R.EmbeddingCallError) as embedding_info:
                await learn(method, [alf_item("t" + "a" * 20, Q1)])
            assert method.entries == ()
        finally:
            await method.close()
    assert embedding_info.value.failure.kind is FailureKind.INFRA
    assert embedding_info.value.failure.role == "embedding"


# --- the H6 contract with memory that actually accumulates -------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark", protocol.BENCHMARKS)
async def test_the_contract_holds_with_a_growing_bank(tmp_path, benchmark):
    items = contract_items(benchmark)
    router = Router(
        judge=[judged("success"), judged("failure")] * 2,
        learner=[items_reply(f"Item {i}") for i in range(4)],
    )
    fake = FakeUpstream(responder=router)
    async with running_gateway(tmp_path, fake) as (gw, _client):
        transport = httpx.ASGITransport(app=gw.app)
        report = await exercise_method(
            lambda: R.ReasoningBank(transport=transport),
            rb_cell(tmp_path, benchmark=benchmark),
            items=items,
            forbidden_strings=["ZX-GOLD-ANSWER"],
        )
    assert len(report.snapshot["entries"]) == len(items)
    assert report.resumed_snapshot == report.snapshot
    assert all(not i.empty for i in report.test_injections)
    assert report.learning_injections[0] == Injection()


# --- learning checkpoints (contract item 6) ------------------------------------------------------


@pytest.mark.asyncio
async def test_a_learning_checkpoint_restores_the_bank_vectors_and_counts(tmp_path):
    """Checkpointed after two items, the bank goes on learning a third, then dies. A
    learning setup with ``cell.resume`` on the checkpoint's work_dir copy rebuilds the
    entries and their vectors, the counters and J's counts, and retrieves exactly as the
    bank did at the checkpoint."""
    import dataclasses
    import shutil

    from evals.skill_bench.methods.base import (
        CheckpointMismatch,
        LearningResume,
        snapshot_sha256,
    )
    from evals.skill_bench.run import _copy_method_dir

    router = Router(
        judge=[judged("success"), judged("failure"), judged("success")],
        learner=[items_reply("A1", "A2"), items_reply("B1"), items_reply("C1")],
        vectors=VECTORS,
    )
    fake = FakeUpstream(responder=router)
    work = tmp_path / "work"
    saved = tmp_path / "checkpoint"
    async with running_gateway(tmp_path, fake) as (gw, _client):
        transport = httpx.ASGITransport(app=gw.app)
        method = R.ReasoningBank(transport=transport)
        await method.setup(rb_cell(tmp_path))
        try:
            await learn(
                method, [alf_item("t" + "a" * 20, Q1), alf_item("t" + "b" * 20, Q2)]
            )
            state = await method.checkpoint()
            recorded = await method.snapshot()
            vectors = [list(v) for v in method._vectors]
            _copy_method_dir(work, saved)
            expected = await method.before_task(TaskView("t" + "c" * 20, Q3, ()))
            await learn(method, [alf_item("t" + "e" * 20, Q3)])  # discarded
        finally:
            await method.close()
        assert state == {
            "counters": recorded["counters"],
            "judge": recorded["judge"],
            "refusal_streaks": {},
        }
        assert json.loads(json.dumps(state)) == state
        resume = LearningResume(
            "c0001-done", "done", 1, state, snapshot_sha256(recorded)
        )

        def restore() -> None:
            shutil.rmtree(work)
            _copy_method_dir(saved, work)

        restore()
        resumed = R.ReasoningBank(transport=transport)
        await resumed.setup(dataclasses.replace(rb_cell(tmp_path), resume=resume))
        try:
            assert await resumed.snapshot() == recorded
            assert [list(v) for v in resumed._vectors] == vectors
            assert resumed._judge is not None and resumed._judge._calls == 2
            assert resumed._judge_counters is resumed._judge.counters
            again = await resumed.before_task(TaskView("t" + "c" * 20, Q3, ()))
        finally:
            await resumed.close()
        assert again == expected

        # A bank file that differs from the checkpoint's is refused.
        restore()
        bank = work / R.BANK_FILE
        bank.write_text(bank.read_text().replace("A2", "A9"))
        with pytest.raises(CheckpointMismatch, match="hashes to"):
            await R.ReasoningBank(transport=transport).setup(
                dataclasses.replace(rb_cell(tmp_path), resume=resume)
            )
        # So are a frozen bank and a state without J's counts.
        restore()
        (work / R.FROZEN_FILE).write_text("{}")
        with pytest.raises(CheckpointMismatch, match="frozen"):
            await R.ReasoningBank(transport=transport).setup(
                dataclasses.replace(rb_cell(tmp_path), resume=resume)
            )
        restore()
        broken = LearningResume(
            "c0001-done",
            "done",
            1,
            {"counters": state["counters"]},
            resume.snapshot_sha256,
        )
        with pytest.raises(CheckpointMismatch, match="cannot be restored"):
            await R.ReasoningBank(transport=transport).setup(
                dataclasses.replace(rb_cell(tmp_path), resume=broken)
            )


# --- refusals: recorded, not stopped (rule harness-stop-learning) -------------------------------

PAYLOAD_413 = Reply(
    413,
    {"error": {"code": 413, "message": "Request payload too large"}},
)
PROVIDER_400 = Reply(
    400, {"error": {"code": 400, "message": "Provider returned error"}}
)
#: The counters every snapshot carried before refusals were counted.
PRE_REFUSAL_COUNTERS = {
    "tasks_learned",
    "judged_success",
    "judged_failure",
    "extractions",
    "extraction_model_failures",
    "extraction_no_reply",
    "extraction_inputs_cut",
    "unparseable_extractions",
    "incomplete_items",
    "items_stored",
    "items_dropped_over_limit",
    "entries",
    "empty_queries",
    "steps_skipped",
    "retrievals",
    "retrievals_with_memory",
}


def test_rb_refusal_counters_stay_out_of_a_snapshot_without_refusals():
    counters = R.RbCounters()
    assert set(counters.to_dict()) == PRE_REFUSAL_COUNTERS
    assert set(R.REFUSAL_COUNTERS) == {"extraction_rejected", "judge_rejected"}
    counters.extraction_rejected = 2
    assert set(counters.to_dict()) == PRE_REFUSAL_COUNTERS | {"extraction_rejected"}
    assert R.RbCounters(**counters.to_dict()) == counters
    old = {key: 1 for key in PRE_REFUSAL_COUNTERS}
    assert R.RbCounters(**old).to_dict() == old


@pytest.mark.asyncio
async def test_a_refused_extraction_stores_no_entry_and_learning_goes_on(tmp_path):
    router = Router(
        judge=[judged("success"), judged("success")],
        learner=[PAYLOAD_413, items_reply("B1")],
        vectors=VECTORS,
    )
    async with running_gateway(tmp_path, FakeUpstream(responder=router)) as (gw, _):
        method = R.ReasoningBank(transport=httpx.ASGITransport(app=gw.app))
        await method.setup(rb_cell(tmp_path))
        try:
            await learn(method, [alf_item("t" + "a" * 20, Q1)])
            assert method.entries == ()
            await learn(method, [alf_item("t" + "b" * 20, Q2)])
            snapshot = await method.snapshot()
            state = await method.checkpoint()
        finally:
            await method.close()
    assert [e["task_uid"] for e in snapshot["entries"]] == ["t" + "b" * 20]
    c = snapshot["counters"]
    assert (c["extractions"], c["extraction_rejected"], c["entries"]) == (2, 1, 1)
    assert c["extraction_no_reply"] == 0 and c["extraction_model_failures"] == 0
    assert c["unparseable_extractions"] == 0 and "judge_rejected" not in c
    rows = read_jsonl(tmp_path / "work" / R.EXTRACTIONS_FILE)
    assert rows[0]["refused"]["status"] == 413 and rows[0]["stored"] is False
    assert rows[0]["http_status"] == 413 and rows[0]["reply"] is None
    assert "refused" not in rows[1] and rows[1]["stored"] is True
    assert state["refusal_streaks"] == {}  # the served extraction ended the streak
    # No embedding call for the refused extraction's entry (only for item b).
    assert [Q1] not in router.embedded() and [Q2] in router.embedded()


@pytest.mark.asyncio
async def test_a_refused_verdict_skips_the_rb_item(tmp_path):
    router = Router(
        judge=[PROVIDER_400, judged("success")],
        learner=[items_reply("B1")],
        vectors=VECTORS,
    )
    async with running_gateway(tmp_path, FakeUpstream(responder=router)) as (gw, _):
        method = R.ReasoningBank(transport=httpx.ASGITransport(app=gw.app))
        await method.setup(rb_cell(tmp_path))
        try:
            await learn(
                method, [alf_item("t" + "a" * 20, Q1), alf_item("t" + "b" * 20, Q2)]
            )
            snapshot = await method.snapshot()
        finally:
            await method.close()
    c = snapshot["counters"]
    assert c["judge_rejected"] == 1 and c["tasks_learned"] == 1
    assert c["extractions"] == 1 and "extraction_rejected" not in c
    assert len(router.calls["learner"]) == 1
    rows = read_jsonl(tmp_path / "work" / R.EXTRACTIONS_FILE)
    assert [r["event"] for r in rows] == ["judge_refused", "extraction"]
    assert rows[0]["task_uid"] == "t" + "a" * 20
    assert rows[0]["failure"]["role"] == "judge" and rows[0]["http_status"] == 400


@pytest.mark.asyncio
async def test_three_refused_extractions_in_a_row_stop(tmp_path):
    from evals.skill_bench.infra import failures as F

    router = Router(
        judge=[judged("success")] * 3,
        learner=[PAYLOAD_413, PAYLOAD_413, PAYLOAD_413],
        vectors=VECTORS,
    )
    items = [alf_item(f"t{c * 20}", Q1) for c in "abc"]
    async with running_gateway(tmp_path, FakeUpstream(responder=router)) as (gw, _):
        method = R.ReasoningBank(transport=httpx.ASGITransport(app=gw.app))
        await method.setup(rb_cell(tmp_path))
        try:
            # The third refusal's after_task holds the stop; the method's
            # next hook (here finalize, after the last item) raises it.
            await learn(method, items)
            snapshot = await method.snapshot()
            with pytest.raises(F.GatewayCallError) as info:
                await method.finalize()
        finally:
            await method.close()
    stop = info.value.failure
    assert (stop.kind, stop.code, stop.role, stop.status) == (
        FailureKind.HARNESS,
        F.REFUSAL_STREAK,
        "learner",
        413,
    )
    assert snapshot["counters"]["extraction_rejected"] == 3
    rows = read_jsonl(tmp_path / "work" / R.EXTRACTIONS_FILE)
    assert len(rows) == 3 and all(r["refused"]["status"] == 413 for r in rows)


@pytest.mark.asyncio
async def test_three_refused_rb_verdicts_in_a_row_stop(tmp_path):
    from evals.skill_bench.infra import failures as F

    router = Router(judge=[PROVIDER_400] * 3, vectors=VECTORS)
    items = [alf_item(f"t{c * 20}", Q1) for c in "abc"]
    async with running_gateway(tmp_path, FakeUpstream(responder=router)) as (gw, _):
        method = R.ReasoningBank(transport=httpx.ASGITransport(app=gw.app))
        await method.setup(rb_cell(tmp_path))
        try:
            await learn(method, items)  # after_task holds the stop
            with pytest.raises(F.GatewayCallError) as info:
                await method.before_task(copy.deepcopy(items[0].view))
        finally:
            await method.close()
    assert info.value.failure.code == F.REFUSAL_STREAK
    assert info.value.failure.role == "judge"
    assert router.calls["learner"] == []


@pytest.mark.asyncio
async def test_an_rb_checkpoint_of_the_code_before_refusals_restores_as_recorded(
    tmp_path,
):
    import dataclasses

    from evals.skill_bench.methods.base import LearningResume, snapshot_sha256

    router = Router(
        judge=[judged("success")] * 3,
        learner=[items_reply("A1"), PAYLOAD_413, PAYLOAD_413],
        vectors=VECTORS,
    )
    async with running_gateway(tmp_path, FakeUpstream(responder=router)) as (gw, _):
        transport = httpx.ASGITransport(app=gw.app)
        method = R.ReasoningBank(transport=transport)
        await method.setup(rb_cell(tmp_path))
        try:
            await learn(method, [alf_item("t" + "a" * 20, Q1)])
            state = await method.checkpoint()
            recorded = await method.snapshot()
            await learn(method, [alf_item("t" + "b" * 20, Q2)])
            refused_state = await method.checkpoint()
            refused_snapshot = await method.snapshot()
        finally:
            await method.close()
        del state["refusal_streaks"]  # as the code before refusals wrote it
        assert set(recorded["counters"]) == PRE_REFUSAL_COUNTERS
        resume = LearningResume(
            "c0001-done", "done", 1, state, snapshot_sha256(recorded)
        )
        resumed = R.ReasoningBank(transport=transport)
        await resumed.setup(dataclasses.replace(rb_cell(tmp_path), resume=resume))
        try:
            assert await resumed.snapshot() == recorded
            assert await resumed.checkpoint() == {**state, "refusal_streaks": {}}
        finally:
            await resumed.close()
        # A checkpoint after a refusal restores its counters and its streak.
        assert refused_state["refusal_streaks"] == {
            "learner": {"signature": "learner/payload/413", "count": 1}
        }
        resume = LearningResume(
            "c0002-done", "done", 2, refused_state, snapshot_sha256(refused_snapshot)
        )
        again = R.ReasoningBank(transport=transport)
        await again.setup(dataclasses.replace(rb_cell(tmp_path), resume=resume))
        try:
            assert await again.snapshot() == refused_snapshot
            assert (await again.checkpoint())["refusal_streaks"] == refused_state[
                "refusal_streaks"
            ]
        finally:
            await again.close()
