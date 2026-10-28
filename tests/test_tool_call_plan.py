"""Which tool calls of a model turn run, and how the agent answers the rest.

Offline and pure: :func:`mirix.agent.tool_call_plan.plan_tool_calls` on its own,
then ``Agent._handle_ai_response`` / ``Agent._get_ai_reply`` on a stub agent
(``Agent.__new__``, no database, no model) whose tool runs are recorded:

* a finish (``finish_memory_update``) runs after the other calls of its
  turn, once; it used to run first and then fail the "must be the last tool
  call" assertion (after it had already run). A reply (``send_message``) is
  content: every distinct one is sent, in order; one that another call of its
  turn follows waits for a later turn (the chain goes on), so the model reads
  those results before it replies;
* the finish-alone escalation runs the finish after its turn's other calls
  only when they went through (a rejected, refused or failed call is read
  first);
* with a per-turn cap only the first calls run, a terminal call only
  alone, every other call is answered "Not run", in the answer's order;
* a call its tool refuses (``ToolCallRefused``) is answered with the
  refusal, counted neither as executed nor as failed, and the chain goes on;
* an agent's ``tool_allowlist`` limits the tools a step offers and runs
  (None: every tool, as before);
* a validation error reaches the interface as an ``Error: `` message.
"""

import json
import logging
from types import SimpleNamespace

import pytest

from mirix.agent.tool_call_plan import (
    FINISH_TOOLS,
    REPLY_TOOLS,
    TERMINAL_ALONE_REMINDERS,
    TERMINAL_TOOLS,
    held_terminal_text,
    plan_tool_calls,
)
from mirix.schemas.enums import ToolType
from mirix.schemas.openai.chat_completion_response import (
    FunctionCall,
    ToolCall,
)
from mirix.schemas.openai.chat_completion_response import (
    Message as ChatCompletionMessage,
)

# --------------------------------------------------------------------------- #
# The plan (pure)                                                              #
# --------------------------------------------------------------------------- #


class TestPlan:
    def test_without_a_cap_every_call_runs_and_a_terminal_call_runs_last(self):
        plan = plan_tool_calls(["finish_memory_update", "experience_triage"])
        assert plan.order == [1, 0] and plan.skipped == {} and not plan.deferred
        plan = plan_tool_calls(["search_in_memory", "send_message"])
        assert plan.order == [0, 1] and plan.skipped == {} and not plan.deferred
        plan = plan_tool_calls(["a", "b", "c"])
        assert plan.order == [0, 1, 2] and plan.skipped == {}
        assert plan.guarded == ()

    def test_a_reply_that_another_call_follows_waits_for_a_later_turn(self):
        """[send_message("Let me look that up"), search]: the search runs, the
        reply is not sent and the chain goes on, so the model answers with the
        search result (it used to be sent last and end the chain)."""
        plan = plan_tool_calls(["send_message", "search_in_memory"])
        assert plan.order == [1] and plan.deferred
        assert plan.skipped == {
            0: "Not run: send_message ends your turn, and this turn called "
            "search_in_memory after it. Read those results, then send your reply."
        }
        # A finish waits too: the chain must not end with a reply unsent.
        plan = plan_tool_calls(["send_message", "skill_list", "finish_memory_update"])
        assert plan.order == [1] and plan.deferred
        assert "ends your turn" in plan.skipped[0]
        assert "waits for a later turn" in plan.skipped[2]

    def test_every_distinct_reply_is_sent_in_order(self):
        parts = ['{"message": "Part 1"}', '{"message": "Part 2"}']
        plan = plan_tool_calls(["send_message", "send_message"], arguments=parts)
        assert plan.order == [0, 1] and plan.skipped == {} and not plan.deferred
        plan = plan_tool_calls(
            ["search_in_memory", "send_message", "send_message"],
            arguments=["{}"] + parts,
        )
        assert plan.order == [0, 1, 2] and plan.skipped == {}
        # Without the arguments no reply is taken for a repeat.
        plan = plan_tool_calls(["send_message", "send_message"])
        assert plan.order == [0, 1]

    def test_an_exact_repeat_of_a_reply_runs_once(self):
        same = ['{"message": "Hi", "x": 1}', '{"x": 1,  "message": "Hi"}']
        plan = plan_tool_calls(["send_message", "send_message"], arguments=same)
        assert plan.order == [0] and not plan.deferred
        assert plan.skipped == {
            1: "Not run: an identical send_message call of this turn ran; it ran once."
        }
        # A waiting reply that the turn's ending reply repeats is not deferred.
        plan = plan_tool_calls(
            ["send_message", "search_in_memory", "send_message"],
            arguments=['{"message": "Hi"}', "{}", '{"message": "Hi"}'],
        )
        assert plan.order == [1, 2] and not plan.deferred
        assert "identical" in plan.skipped[0]

    def test_a_repeated_terminal_call_runs_once(self):
        plan = plan_tool_calls(["finish_memory_update", "finish_memory_update"])
        assert plan.order == [1]
        assert plan.skipped == {
            0: "Not run: finish_memory_update was called more than once in this "
            "turn; it ran once."
        }
        assert not plan.deferred  # the chain ends as the finish says
        # A reply and a finish both run: the reply is content, the finish last.
        plan = plan_tool_calls(["skill_list", "send_message", "finish_memory_update"])
        assert plan.order == [0, 1, 2] and plan.skipped == {}

    def test_a_cap_runs_the_first_calls_only(self):
        plan = plan_tool_calls(["a", "b", "c"], max_calls=1)
        assert plan.order == [0] and set(plan.skipped) == {1, 2}
        assert plan.deferred
        assert plan.skipped[1] == (
            "Not run: this step takes one tool call per turn (a ran), so you read "
            "each result before your next call. Call b again if you still need it."
        )
        plan = plan_tool_calls(["a", "b", "c"], max_calls=2)
        assert plan.order == [0, 1] and set(plan.skipped) == {2}
        assert "at most 2 tool calls per turn (a, b ran)" in plan.skipped[2]

    def test_with_a_cap_a_terminal_call_runs_only_alone(self):
        plan = plan_tool_calls(
            ["experience_triage", "finish_memory_update"], max_calls=1
        )
        assert plan.order == [0] and plan.deferred
        assert plan.skipped == {
            1: "Not run: call finish_memory_update() alone, in a turn of its own, "
            "after reading the results of your other calls."
        }
        plan = plan_tool_calls(["finish_memory_update"], max_calls=1)
        assert plan.order == [0] and plan.skipped == {} and not plan.deferred
        # Two finishes and nothing else: the finish still happens, once.
        plan = plan_tool_calls(
            ["finish_memory_update", "finish_memory_update"], max_calls=1
        )
        assert plan.order == [1] and set(plan.skipped) == {0}

    def test_a_terminal_call_paired_again_runs_after_the_others(self):
        """A model told once to call finish alone that pairs it again is not
        held until the step cap: with ``terminal_after_others`` the terminal
        call runs last, after the other calls, when all of them fit the cap;
        otherwise it waits as before (it never ends a turn that left a call
        unrun)."""
        plan = plan_tool_calls(
            ["experience_triage", "finish_memory_update"],
            max_calls=1,
            terminal_after_others=True,
        )
        assert plan.order == [0, 1] and plan.skipped == {}
        assert not plan.deferred and not plan.terminal_deferred
        # The agent runs it only when the triage went through.
        assert plan.guarded == (1,)
        plan = plan_tool_calls(
            ["skill_list", "skill_read", "finish_memory_update"],
            max_calls=1,
            terminal_after_others=True,
        )
        assert plan.order == [0] and set(plan.skipped) == {1, 2}
        assert plan.deferred and plan.terminal_deferred and plan.guarded == ()
        # Without it (the first time), the terminal call waits for a turn alone.
        plan = plan_tool_calls(
            ["experience_triage", "finish_memory_update"], max_calls=1
        )
        assert plan.terminal_deferred and plan.order == [0]
        assert TERMINAL_ALONE_REMINDERS == 1

    def test_a_burst_runs_one_call(self):
        """One answer calling every tool once."""
        names = [
            "skill_list",
            "skill_read",
            "skill_create",
            "skill_edit",
            "skill_delete",
            "experience_triage",
            "finish_memory_update",
        ]
        plan = plan_tool_calls(names, max_calls=1)
        assert plan.order == [0] and sorted(plan.skipped) == [1, 2, 3, 4, 5, 6]

    def test_a_cap_below_one_is_refused(self):
        with pytest.raises(ValueError):
            plan_tool_calls(["a"], max_calls=0)
        assert TERMINAL_TOOLS == ("send_message", "finish_memory_update")
        assert set(FINISH_TOOLS) | set(REPLY_TOOLS) == set(TERMINAL_TOOLS)

    # -- replies of one turn go out together ---------------------------------- #

    def test_when_a_reply_waits_every_reply_of_its_turn_waits(self):
        """[send_message(A), search, send_message(B)] used to send B, defer A
        and force the chain on, so the model replied again: two replies from
        one turn, out of order. Now both wait: the model reads the search
        result and sends its whole reply in a later turn."""
        plan = plan_tool_calls(
            ["send_message", "search_in_memory", "send_message"],
            arguments=['{"message": "A"}', '{"query": "q"}', '{"message": "B"}'],
        )
        assert plan.order == [1] and plan.deferred and not plan.terminal_deferred
        assert plan.skipped == {
            0: "Not run: send_message ends your turn, and this turn called "
            "search_in_memory after it. Read those results, then send your reply.",
            2: "Not run: an earlier send_message of this turn waits for the "
            "results of the calls made after it, so this send_message waits too "
            "(a turn's replies go out together). Read those results, then send "
            "your reply.",
        }
        # Several ending replies, a repeat among them, and a finish: none runs
        # (nothing is answered "ran once" when nothing ran), the finish waits.
        plan = plan_tool_calls(
            [
                "send_message",
                "search_in_memory",
                "send_message",
                "send_message",
                "finish_memory_update",
            ],
            arguments=[
                '{"message": "A"}',
                "{}",
                '{"message": "B"}',
                '{"message": "B"}',
                "{}",
            ],
        )
        assert plan.order == [1] and plan.deferred
        assert "ends your turn" in plan.skipped[0]
        assert "waits too" in plan.skipped[2] and "waits too" in plan.skipped[3]
        assert "waits for a later turn" in plan.skipped[4]
        assert not any("ran once" in text for text in plan.skipped.values())
        # Unchanged: a waiting reply that the ending reply repeats is not a
        # second reply (the reply is sent once, test above), and replies that
        # end the turn run when none waits.
        plan = plan_tool_calls(
            ["search_in_memory", "send_message", "send_message"],
            arguments=["{}", '{"message": "A"}', '{"message": "B"}'],
        )
        assert plan.order == [0, 1, 2] and not plan.deferred

    # -- a terminal name the agent cannot call --------------------------------- #

    def test_a_terminal_name_the_agent_cannot_call_is_an_ordinary_call(self):
        """A memory sub-agent (no per-turn cap) has finish_memory_update but no
        send_message. A model that hallucinates send_message in
        [send_message, episodic_memory_insert, finish_memory_update] got the
        finish deferred and was told to "send your reply" with a tool it does
        not have. The call is an ordinary one: it runs (the agent answers it
        "not available"), and the finish still runs, last."""
        memory_agent = {"episodic_memory_insert", "finish_memory_update"}
        names = ["send_message", "episodic_memory_insert", "finish_memory_update"]
        plan = plan_tool_calls(names, available=memory_agent)
        assert plan.order == [0, 1, 2] and plan.skipped == {}
        assert not plan.deferred and not plan.terminal_deferred
        assert plan.guarded == ()
        # Without the names every terminal name counts as callable (as before).
        assert plan_tool_calls(names).order == [1]
        # A hallucinated finish is ordinary too: it never ends the turn early.
        plan = plan_tool_calls(
            ["send_message", "finish_memory_update", "search_in_memory"],
            available={"send_message", "search_in_memory"},
        )
        assert plan.order == [1, 2] and plan.deferred
        assert "ends your turn" in plan.skipped[0]
        assert "search_in_memory" in plan.skipped[0]
        # Only the call it can make is what the reply waits for.
        assert "finish_memory_update" not in plan.skipped[0]

    def test_the_curator_is_never_told_to_call_a_tool_it_lacks_alone(self):
        """The skill curator (one call per turn, SKILL_CURATOR_TOOLS, no
        send_message): [skill_edit, send_message] was answered "call
        send_message() alone"; the next call then failed "not available" and
        spent a step of the hard-capped budget. send_message is answered "not
        available" at once (it runs nothing, so it is never asked for
        again)."""
        from mirix.constants import SKILL_CURATOR_TOOLS

        assert "send_message" not in SKILL_CURATOR_TOOLS
        plan = plan_tool_calls(
            ["skill_edit", "send_message"],
            max_calls=1,
            available=SKILL_CURATOR_TOOLS,
        )
        assert plan.order == [0, 1] and plan.skipped == {}
        assert not plan.deferred and not plan.terminal_deferred
        # Past the cap's window it is answered at once; a call the step can
        # make waits ("Call it again").
        plan = plan_tool_calls(
            ["skill_list", "send_message", "skill_read"],
            max_calls=1,
            available=SKILL_CURATOR_TOOLS,
        )
        assert plan.order == [0, 1] and set(plan.skipped) == {2}
        assert "one tool call per turn (skill_list ran)" in plan.skipped[2]
        # Any name it lacks, not only a terminal one.
        plan = plan_tool_calls(
            ["skill_list", "web_search"], max_calls=1, available=SKILL_CURATOR_TOOLS
        )
        assert plan.order == [0, 1] and plan.skipped == {}
        # The window takes the turn's first calls whatever they call: a burst
        # led by a call the step cannot make runs nothing real.
        plan = plan_tool_calls(
            ["web_search", "skill_create", "send_message"],
            max_calls=1,
            available=SKILL_CURATOR_TOOLS,
        )
        assert plan.order == [0, 2] and set(plan.skipped) == {1}
        assert plan.deferred and not plan.terminal_deferred

    def test_an_unavailable_call_is_still_an_other_call_for_a_real_finish(self):
        """It is an ordinary call: with a cap a real finish paired with it waits
        for a turn alone (the model reads the "not available" answer first),
        and once reminded the finish runs after it, guarded (the agent holds
        the finish because that call failed)."""
        from mirix.constants import SKILL_CURATOR_TOOLS

        names = ["send_message", "finish_memory_update"]
        plan = plan_tool_calls(names, max_calls=1, available=SKILL_CURATOR_TOOLS)
        assert plan.order == [0] and plan.terminal_deferred
        assert "call finish_memory_update() alone" in plan.skipped[1]
        plan = plan_tool_calls(
            ["skill_edit", "send_message", "finish_memory_update"],
            max_calls=1,
            terminal_after_others=True,
            available=SKILL_CURATOR_TOOLS,
        )
        assert plan.order == [0, 1, 2] and plan.guarded == (2,)
        assert plan.skipped == {} and not plan.deferred
        # Without a cap, an unavailable call after a reply does not make it
        # wait: it can only be answered "not available", so it has no result
        # the reply could use. It is answered first.
        plan = plan_tool_calls(
            ["send_message", "web_search"], available={"send_message"}
        )
        assert plan.order == [1, 0] and plan.skipped == {} and not plan.deferred

    def test_an_unavailable_terminal_name_takes_no_slot_of_the_cap(self):
        """The curator's [send_message, skill_edit] (one call per
        turn, no send_message) filled the window with the unavailable
        send_message, answered skill_edit "(send_message ran) ... Call
        skill_edit again", and a model that keeps leading with send_message
        never ran its real call. A reply or a finish the step cannot make is
        no step later calls build on: it is answered "not available" and the
        cap's window starts after it. Other unknown names still take their
        slot (a burst led by one runs nothing real)."""
        from mirix.constants import SKILL_CURATOR_TOOLS

        plan = plan_tool_calls(
            ["send_message", "skill_edit"],
            max_calls=1,
            available=SKILL_CURATOR_TOOLS,
        )
        assert plan.order == [0, 1] and plan.skipped == {} and not plan.deferred
        plan = plan_tool_calls(
            ["send_message", "skill_edit", "send_message", "skill_read"],
            max_calls=1,
            arguments=['{"message": "a"}', "{}", '{"message": "b"}', "{}"],
            available=SKILL_CURATOR_TOOLS,
        )
        assert plan.order == [0, 1, 2] and set(plan.skipped) == {3}
        assert "(skill_edit ran)" in plan.skipped[3] and plan.deferred
        # A chat step with a cap that lacks the finish: a hallucinated finish
        # takes no slot either.
        chat = {"send_message", "search_in_memory"}
        plan = plan_tool_calls(
            ["finish_memory_update", "search_in_memory", "send_message"],
            max_calls=1,
            available=chat,
        )
        assert plan.order == [0, 1] and plan.terminal_deferred
        assert "call send_message() alone" in plan.skipped[2]

    def test_a_capped_out_call_is_never_told_an_unavailable_call_ran(self):
        """The "Not run" answer of a call past the cap names the calls that
        ran, and the calls in the window the step cannot make as "not
        available" — never as run."""
        from mirix.constants import SKILL_CURATOR_TOOLS

        plan = plan_tool_calls(
            ["web_search", "skill_edit"], max_calls=1, available=SKILL_CURATOR_TOOLS
        )
        assert plan.order == [0] and set(plan.skipped) == {1}
        assert "ran" not in plan.skipped[1].split("(", 1)[1].split(")", 1)[0]
        assert "(web_search is not available)" in plan.skipped[1]
        assert "Call skill_edit again" in plan.skipped[1]
        plan = plan_tool_calls(
            ["skill_list", "web_search", "no_such_tool", "skill_read"],
            max_calls=3,
            available=SKILL_CURATOR_TOOLS,
        )
        assert plan.order == [0, 1, 2] and set(plan.skipped) == {3}
        assert (
            "(skill_list ran; web_search, no_such_tool are not available)"
            in plan.skipped[3]
        )

    def test_an_unavailable_call_never_makes_a_reply_wait(self):
        """A chat agent (send_message, no finish) answering
        [send_message(A), finish_memory_update] had A held back "Read those
        results, then send your reply" for a call that could only fail. The
        reply waits only for calls the agent can make; the others are
        answered "not available" before it."""
        chat = {"send_message", "search_in_memory"}
        plan = plan_tool_calls(
            ["send_message", "finish_memory_update"],
            arguments=['{"message": "A"}', "{}"],
            available=chat,
        )
        assert plan.order == [1, 0] and plan.skipped == {} and not plan.deferred
        plan = plan_tool_calls(
            ["search_in_memory", "send_message", "ghost"],
            arguments=["{}", '{"message": "A"}', "{}"],
            available=chat,
        )
        assert plan.order == [0, 2, 1] and not plan.deferred
        # A call it can make after the reply still makes it wait, and the
        # answer names only that call.
        plan = plan_tool_calls(
            ["send_message", "ghost", "search_in_memory", "finish_memory_update"],
            arguments=['{"message": "A"}', "{}", "{}", "{}"],
            available=chat,
        )
        assert plan.order == [1, 2, 3] and set(plan.skipped) == {0}
        assert "this turn called search_in_memory after it" in plan.skipped[0]
        assert plan.deferred

    @pytest.mark.parametrize("cap", [None, 1, 2])
    @pytest.mark.parametrize(
        "name",
        ["send_message", "finish_memory_update", "skill_list", "no_such_tool", ""],
    )
    @pytest.mark.parametrize(
        "available",
        [None, (), ("skill_list",), ("send_message", "finish_memory_update")],
    )
    def test_a_single_call_turn_is_planned_as_before(self, cap, name, available):
        """One call: the plan never depends on what the agent can call."""
        plan = plan_tool_calls([name], max_calls=cap, available=available)
        assert plan == plan_tool_calls([name], max_calls=cap)
        assert plan.order == [0] and plan.skipped == {} and not plan.deferred


# --------------------------------------------------------------------------- #
# The agent's handling of a turn (a stub agent, no database)                   #
# --------------------------------------------------------------------------- #


class _Interface:
    def __init__(self):
        self.function_messages = []

    def internal_monologue(self, *args, **kwargs):
        pass

    def function_message(self, msg, msg_obj=None, **kwargs):
        self.function_messages.append(msg)


def _tool(name):
    # USER_DEFINED: the agent does not filter its arguments by a module function.
    return SimpleNamespace(
        name=name,
        tool_type=ToolType.USER_DEFINED,
        return_char_limit=10_000,
        json_schema={
            "name": name,
            "description": f"The {name} tool.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    )


_TOOL_NAMES = (
    "skill_list",
    "skill_read",
    "experience_triage",
    "finish_memory_update",
    "send_message",
    "search_in_memory",
    "skill_create",
)


def _stub_agent(
    monkeypatch,
    *,
    refuse=(),
    decline=(),
    allowlist=None,
    cap=None,
    tools=_TOOL_NAMES,
):
    """An Agent with no database: each tool run is recorded (its name in
    ``ran``, its arguments in ``agent.ran_args``); a tool named in ``refuse``
    raises ToolCallRefused, one named in ``decline`` rejects the call with an
    answer the way the skill tools do (``declined_tool_calls``). ``tools``:
    the names of the agent's tools."""
    import mirix.agent.agent as agent_module
    from mirix.agent.agent import Agent
    from mirix.errors import ToolCallRefused
    from mirix.helpers import ToolRulesSolver
    from mirix.schemas.agent import AgentType

    monkeypatch.setattr(agent_module, "CLEAR_HISTORY_AFTER_MEMORY_UPDATE", False)
    agent = Agent.__new__(Agent)
    agent.agent_state = SimpleNamespace(
        id="agent-stub",
        name="stub_agent",
        agent_type=AgentType.procedural_memory_agent,
        tools=[_tool(n) for n in tools],
        is_type=lambda *types: False,
        llm_config=SimpleNamespace(model="fake-model", context_window=100_000),
    )
    agent.model = "fake-model"
    agent.interface = _Interface()
    agent.logger = logging.getLogger("test_tool_call_plan")
    agent.tool_rules_solver = ToolRulesSolver(tool_rules=[])
    agent.last_function_response = None
    agent.tool_allowlist = allowlist
    agent._max_tool_calls_per_turn = cap
    ran = []
    agent.ran_args = []

    async def execute(function_name, function_args, target_mirix_tool, **kwargs):
        ran.append(function_name)
        agent.ran_args.append(dict(function_args))
        if function_name in refuse:
            raise ToolCallRefused(f"{function_name} refuses: address the rest first.")
        if function_name in decline:
            agent.declined_tool_calls = getattr(agent, "declined_tool_calls", 0) + 1
            return f"Rejected: {function_name} was not carried out."
        return f"{function_name} done"

    agent.execute_tool_and_persist_state = execute
    return agent, ran


def _turn(*names, arguments=None):
    return ChatCompletionMessage(
        role="assistant",
        tool_calls=[
            ToolCall(
                id=f"call-{i}",
                function=FunctionCall(name=name, arguments=json.dumps(arguments or {})),
            )
            for i, name in enumerate(names)
        ],
    )


def _calls(*calls):
    """A turn of ``(name, arguments)`` tool calls."""
    return ChatCompletionMessage(
        role="assistant",
        tool_calls=[
            ToolCall(
                id=f"call-{i}",
                function=FunctionCall(name=name, arguments=json.dumps(arguments)),
            )
            for i, (name, arguments) in enumerate(calls)
        ],
    )


def _results(messages):
    """``(tool_call_id, status, message)`` of each tool message, in order."""
    out = []
    for message in messages:
        if message.role == "tool":
            body = json.loads(message.content[0].text)
            out.append((message.tool_call_id, body["status"], body["message"]))
    return out


async def _handle(agent, response):
    return await agent._handle_ai_response(
        input_message=SimpleNamespace(session_id=None), response_message=response
    )


@pytest.mark.asyncio
class TestHandleTurn:
    async def test_a_finish_before_another_call_runs_last_without_an_error(
        self, monkeypatch
    ):
        """[finish_memory_update, experience_triage] used to run the finish,
        fail it on the "must be the last tool call" assertion, then run the
        triage: the batch went on after a finish that had already run."""
        agent, ran = _stub_agent(monkeypatch)
        messages, continue_chaining, failed = await _handle(
            agent, _turn("finish_memory_update", "experience_triage")
        )
        assert ran == ["experience_triage", "finish_memory_update"]
        # Tool messages in the answer's order, each call answered once.
        assert [r[0] for r in _results(messages)] == ["call-0", "call-1"]
        assert {r[1] for r in _results(messages)} == {"OK"}
        assert failed is False and continue_chaining is False
        assert agent.last_executed_functions == [
            "experience_triage",
            "finish_memory_update",
        ]
        assert getattr(agent, "failed_tool_calls", 0) == 0

    async def test_a_send_message_runs_after_the_other_calls_once(self, monkeypatch):
        agent, ran = _stub_agent(monkeypatch)
        messages, continue_chaining, failed = await _handle(
            agent, _turn("send_message", "search_in_memory", "send_message")
        )
        assert ran == ["search_in_memory", "send_message"]
        results = _results(messages)
        assert [r[0] for r in results] == ["call-0", "call-1", "call-2"]
        assert results[0][1] == "Failed" and "ran once" in results[0][2]
        assert not failed and continue_chaining is False
        assert agent.skipped_tool_calls == 1 and agent.multi_call_turns == 1

    async def test_distinct_replies_are_all_sent_in_order(self, monkeypatch):
        """[send_message("Part 1"), send_message("Part 2")]: both reach the
        user, in order (only "Part 2" was sent, and "Part 1" was answered
        "called more than once")."""
        agent, ran = _stub_agent(monkeypatch)
        messages, continue_chaining, failed = await _handle(
            agent,
            _calls(
                ("send_message", {"message": "Part 1: the options."}),
                ("send_message", {"message": "Part 2: which one should I book?"}),
            ),
        )
        assert ran == ["send_message", "send_message"]
        assert [a["message"] for a in agent.ran_args] == [
            "Part 1: the options.",
            "Part 2: which one should I book?",
        ]
        assert {r[1] for r in _results(messages)} == {"OK"}
        assert continue_chaining is False and failed is False

    async def test_a_reply_before_a_search_waits_and_the_chain_goes_on(
        self, monkeypatch
    ):
        """[send_message("Let me look that up"), search]: the search runs and
        the chain goes on, so the model can answer with its result (the
        interim reply used to be sent last and end the chain)."""
        agent, ran = _stub_agent(monkeypatch)
        messages, continue_chaining, failed = await _handle(
            agent,
            _calls(
                ("send_message", {"message": "Let me look that up."}),
                ("search_in_memory", {"query": "flights"}),
            ),
        )
        assert ran == ["search_in_memory"]
        results = _results(messages)
        assert [r[0] for r in results] == ["call-0", "call-1"]
        assert results[0][1] == "Failed" and "Read those results" in results[0][2]
        assert results[1][1] == "OK"
        assert continue_chaining is True and failed is False

    async def test_two_replies_around_a_search_both_wait(self, monkeypatch):
        """[send_message(A), search, send_message(B)] sent B,
        deferred A and forced the chain on, so the model replied again. No
        reply of the turn is sent; the search runs and the chain goes on."""
        agent, ran = _stub_agent(monkeypatch)
        messages, continue_chaining, failed = await _handle(
            agent,
            _calls(
                ("send_message", {"message": "Let me check."}),
                ("search_in_memory", {"query": "flights"}),
                ("send_message", {"message": "You have no flights booked."}),
            ),
        )
        assert ran == ["search_in_memory"]
        results = _results(messages)
        assert [r[0] for r in results] == ["call-0", "call-1", "call-2"]
        assert [r[1] for r in results] == ["Failed", "OK", "Failed"]
        assert "Read those results" in results[0][2]
        assert "waits too" in results[2][2]
        assert continue_chaining is True and failed is False
        assert agent.last_executed_functions == ["search_in_memory"]
        assert agent.skipped_tool_calls == 2

    async def test_a_memory_agent_s_hallucinated_reply_never_holds_its_finish(
        self, monkeypatch
    ):
        """A memory sub-agent has no send_message. In
        [send_message, episodic_memory_insert, finish_memory_update] the reply
        was deferred with the finish ("send your reply" with a tool it lacks)
        and the chain ran on until MAX_CHAINING_STEPS. Now the hallucinated
        call is answered "No function named send_message" at once, the insert
        runs and the finish runs last."""
        agent, ran = _stub_agent(
            monkeypatch, tools=("episodic_memory_insert", "finish_memory_update")
        )
        messages, continue_chaining, failed = await _handle(
            agent,
            _calls(
                ("send_message", {"message": "Saved your trip."}),
                ("episodic_memory_insert", {"items": []}),
                ("finish_memory_update", {}),
            ),
        )
        assert ran == ["episodic_memory_insert", "finish_memory_update"]
        results = _results(messages)
        assert results == [
            ("call-0", "Failed", "No function named send_message"),
            ("call-1", "OK", "episodic_memory_insert done"),
            ("call-2", "OK", "finish_memory_update done"),
        ]
        assert agent.last_executed_functions == [
            "episodic_memory_insert",
            "finish_memory_update",
        ]
        # The finish ended the chain; the failed call is one the model reads
        # like any failed call (step() chains once on function_failed).
        assert continue_chaining is False and failed is True
        assert getattr(agent, "skipped_tool_calls", 0) == 0
        [failure] = agent.tool_failures.as_list()
        assert (failure["tool"], failure["stage"]) == ("send_message", "unknown_tool")

    async def test_the_curator_is_answered_not_available_not_call_it_alone(
        self, monkeypatch
    ):
        """The skill curator (one call per turn, its allowlist
        SKILL_CURATOR_TOOLS) answered [skill_edit, send_message] with "call
        send_message() alone", a tool outside its allowlist. Now skill_edit
        runs and send_message gets its "not available" answer in the same
        turn; nothing asks for a turn alone."""
        from mirix.constants import SKILL_CURATOR_TOOLS

        agent, ran = _stub_agent(
            monkeypatch,
            tools=tuple(SKILL_CURATOR_TOOLS) + ("send_message",),
            allowlist=frozenset(SKILL_CURATOR_TOOLS),
            cap=1,
        )
        messages, continue_chaining, failed = await _handle(
            agent,
            _calls(
                (
                    "skill_edit",
                    {
                        "skill_id": "proc-1",
                        "field": "description",
                        "value": "Opens an item and checks it.",
                    },
                ),
                ("send_message", {"message": "Done."}),
            ),
        )
        assert ran == ["skill_edit"]
        results = _results(messages)
        assert results[0] == ("call-0", "OK", "skill_edit done")
        assert results[1][:2] == ("call-1", "Failed")
        assert results[1][2].startswith(
            "Tool send_message is not available in this step"
        )
        assert not any("alone" in r[2] for r in results)
        assert getattr(agent, "_terminal_alone_turns", 0) == 0
        assert continue_chaining is True and failed is True
        assert getattr(agent, "skipped_tool_calls", 0) == 0

    async def test_the_escalated_finish_waits_for_a_rejected_sibling(self, monkeypatch):
        """Cap 1, the model already told to call finish alone: a pair whose
        other call is rejected (an answer, not an error) holds the finish, so
        the model reads the rejection; a clean pair then runs both."""
        agent, ran = _stub_agent(monkeypatch, cap=1, decline={"skill_read"})
        agent._terminal_alone_turns = TERMINAL_ALONE_REMINDERS
        messages, continue_chaining, failed = await _handle(
            agent,
            _calls(("skill_read", {}), ("finish_memory_update", {})),
        )
        assert ran == ["skill_read"]
        results = _results(messages)
        assert results[0][1] == "OK" and results[0][2].startswith("Rejected: ")
        assert results[1] == (
            "call-1",
            "Failed",
            held_terminal_text("finish_memory_update"),
        )
        assert continue_chaining is True and failed is False
        assert agent.last_executed_functions == ["skill_read"]
        # Still a repeat: the next pairing runs both when they go through.
        assert agent._terminal_alone_turns == TERMINAL_ALONE_REMINDERS
        messages, continue_chaining, failed = await _handle(
            agent,
            _calls(("experience_triage", {}), ("finish_memory_update", {})),
        )
        assert ran[1:] == ["experience_triage", "finish_memory_update"]
        assert continue_chaining is False and failed is False

    async def test_the_escalated_finish_waits_for_a_failed_or_refused_sibling(
        self, monkeypatch
    ):
        agent, ran = _stub_agent(monkeypatch, cap=1, refuse={"skill_list"})
        agent._terminal_alone_turns = TERMINAL_ALONE_REMINDERS
        _, continue_chaining, failed = await _handle(
            agent, _calls(("skill_list", {}), ("finish_memory_update", {}))
        )
        assert ran == ["skill_list"] and continue_chaining is True
        # A validation failure (an empty name) holds it too.
        agent2, ran2 = _stub_agent(monkeypatch, cap=1)
        agent2._terminal_alone_turns = TERMINAL_ALONE_REMINDERS
        messages, continue_chaining, failed = await _handle(
            agent2,
            _calls(
                (
                    "skill_create",
                    {
                        "name": "",
                        "description": "x",
                        "instructions": "x",
                        "entry_type": "guide",
                    },
                ),
                ("finish_memory_update", {}),
            ),
        )
        assert ran2 == [] and failed is True and continue_chaining is True
        assert _results(messages)[1][2] == held_terminal_text("finish_memory_update")

    async def test_with_a_cap_one_call_runs_and_every_call_is_answered(
        self, monkeypatch
    ):
        """Six calls in one answer; only the first runs, the others are
        answered "Not run" in the answer's order, and the chain goes on."""
        agent, ran = _stub_agent(monkeypatch, cap=1)
        messages, continue_chaining, failed = await _handle(
            agent,
            _turn(
                "skill_list",
                "skill_read",
                "skill_create",
                "search_in_memory",
                "experience_triage",
                "finish_memory_update",
            ),
        )
        assert ran == ["skill_list"]
        results = _results(messages)
        assert [r[0] for r in results] == [f"call-{i}" for i in range(6)]
        assert results[0][1] == "OK"
        assert all(
            r[1] == "Failed" and r[2].startswith("Not run: ") for r in results[1:]
        )
        assert "call finish_memory_update() alone" in results[5][2]
        assert continue_chaining is True and failed is False
        assert agent.last_executed_functions == ["skill_list"]
        assert agent.skipped_tool_calls == 5
        assert getattr(agent, "failed_tool_calls", 0) == 0

    async def test_a_refused_call_is_answered_and_the_chain_goes_on(self, monkeypatch):
        """finish_memory_update refused by its tool: not executed, not
        failed, and the model gets the turn back."""
        agent, ran = _stub_agent(monkeypatch, refuse={"finish_memory_update"})
        messages, continue_chaining, failed = await _handle(
            agent, _turn("finish_memory_update")
        )
        assert ran == ["finish_memory_update"]
        [(call_id, status, text)] = _results(messages)
        assert status == "Failed" and "refuses: address the rest first" in text
        assert continue_chaining is True and failed is False
        assert agent.last_executed_functions == []
        assert getattr(agent, "failed_tool_calls", 0) == 0

    async def test_an_allowlist_refuses_a_tool_outside_it_without_running_it(
        self, monkeypatch
    ):
        """A tool the agent has, but that this step may not use."""
        agent, ran = _stub_agent(
            monkeypatch, allowlist=frozenset({"skill_list", "finish_memory_update"})
        )
        messages, continue_chaining, failed = await _handle(
            agent, _turn("search_in_memory")
        )
        assert ran == []
        [(call_id, status, text)] = _results(messages)
        assert status == "Failed"
        assert text == (
            "Tool search_in_memory is not available in this step; available: "
            "skill_list, finish_memory_update."
        )
        assert failed is True and agent.failed_tool_calls == 1
        [failure] = agent.tool_failures.as_list()
        assert (failure["tool"], failure["stage"]) == (
            "search_in_memory",
            "unknown_tool",
        )

    async def test_arguments_that_do_not_fit_the_signature_are_the_models(
        self, monkeypatch
    ):
        """A model leaving out a required argument made Python
        raise TypeError inside the call, recorded as MIRIX's own execution
        failure. The arguments are bound first: a validation failure, the
        model's mistake, and the tool never runs."""
        from mirix.agent import agent as agent_module
        from mirix.errors import ToolArgumentError

        ran = []

        async def needs_field(self, skill_id: str, field: str) -> str:
            ran.append((skill_id, field))
            return "edited"

        try:
            agent_module._bind_tool_arguments(
                "skill_edit", needs_field, {"self": None, "skill_id": "s"}
            )
        except ToolArgumentError as exc:
            assert "skill_edit(): missing a required argument: 'field'" in str(exc)
        else:  # pragma: no cover - the check must raise
            raise AssertionError("no ToolArgumentError")
        agent_module._bind_tool_arguments(
            "skill_edit", needs_field, {"self": None, "skill_id": "s", "field": "f"}
        )
        assert ran == []

        # In a step: the error result is recorded as a validation failure.
        agent, _ = _stub_agent(monkeypatch)

        async def execute(function_name, function_args, target_mirix_tool, **kwargs):
            agent.last_tool_exception = ToolArgumentError(
                "skill_list(): missing a required argument: 'query'"
            )
            return "Error: skill_list(): missing a required argument: 'query'"

        agent.execute_tool_and_persist_state = execute
        _, _, failed = await _handle(agent, _turn("skill_list"))
        assert failed is True
        [failure] = agent.tool_failures.as_list()
        assert (failure["stage"], failure["exception"], failure["argument_error"]) == (
            "validation",
            "ToolArgumentError",
            True,
        )

    async def test_a_real_tool_run_binds_its_arguments_first(self, monkeypatch):
        """execute_tool_and_persist_state (no Langfuse, no database): a memory
        tool called without its required argument is answered with an error
        naming the argument, never run, and its exception is ToolArgumentError."""
        from mirix.errors import ToolArgumentError
        from mirix.functions.function_sets import memory_tools

        agent, _ = _stub_agent(monkeypatch)
        # Undo the stub's recorder: this test runs the real tool dispatch.
        del agent.execute_tool_and_persist_state

        async def no_blocks(**kwargs):
            return []

        agent.block_manager = SimpleNamespace(get_blocks=no_blocks)
        agent.user = SimpleNamespace(timezone="UTC")
        agent._block_scopes = None
        ran = []

        async def fake_skill_read(self, name_or_id: str) -> str:
            ran.append(name_or_id)
            return "the skill"

        monkeypatch.setattr(memory_tools, "skill_read", fake_skill_read)
        tool = SimpleNamespace(
            name="skill_read",
            tool_type=ToolType.MIRIX_MEMORY_CORE,
            return_char_limit=10_000,
        )
        answer = await agent.execute_tool_and_persist_state("skill_read", {}, tool)
        assert ran == [] and "name_or_id" in answer
        assert isinstance(agent.last_tool_exception, ToolArgumentError)
        answer = await agent.execute_tool_and_persist_state(
            "skill_read", {"name_or_id": "proc-1"}, tool
        )
        assert ran == ["proc-1"] and agent.last_tool_exception is None

    async def test_a_validation_error_reaches_the_interface_as_an_error(
        self, monkeypatch
    ):
        """The interface used to get "Validation Error: ...", a prefix no
        interface recognizes."""
        agent, ran = _stub_agent(monkeypatch)
        await _handle(
            agent,
            _turn(
                "skill_create",
                arguments={
                    "name": "",
                    "description": "x",
                    "instructions": "x",
                    "entry_type": "guide",
                },
            ),
        )
        assert ran == []
        errors = [m for m in agent.interface.function_messages if "alidation" in m]
        assert errors == ["Error: Validation error: 'name' cannot be empty."]


# --------------------------------------------------------------------------- #
# What a step offers the model                                                 #
# --------------------------------------------------------------------------- #


class _RecordingClient:
    def __init__(self):
        self.tools = None
        self.llm_config = SimpleNamespace(model="fake-model")

    async def send_llm_request(self, *, messages, tools, **kwargs):
        from mirix.schemas.openai.chat_completion_response import (
            ChatCompletionResponse,
            Choice,
            UsageStatistics,
        )

        self.tools = [t["name"] for t in tools]
        return ChatCompletionResponse(
            id="r1",
            choices=[
                Choice(
                    finish_reason="tool_calls",
                    index=0,
                    message=_turn("finish_memory_update"),
                )
            ],
            created=0,
            model="fake-model",
            usage=UsageStatistics(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


@pytest.mark.asyncio
class TestOfferedTools:
    async def _offered(self, agent):
        client = _RecordingClient()
        agent.supports_structured_output = True
        agent._auto_summarize = False
        await agent._get_ai_reply(message_sequence=[], llm_client=client)
        return client.tools

    async def test_without_an_allowlist_every_tool_is_offered(self, monkeypatch):
        agent, _ = _stub_agent(monkeypatch)
        assert await self._offered(agent) == list(_TOOL_NAMES)
        assert [t.name for t in agent._visible_tools()] == list(_TOOL_NAMES)

    async def test_an_allowlist_offers_only_its_tools(self, monkeypatch):
        agent, _ = _stub_agent(
            monkeypatch, allowlist=frozenset({"skill_list", "finish_memory_update"})
        )
        assert await self._offered(agent) == ["skill_list", "finish_memory_update"]


# --------------------------------------------------------------------------- #
# The request hint (optional): parallel_tool_calls only when configured        #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
class TestParallelToolCallsHint:
    async def _request(self, *, parallel, tools):
        from mirix.llm_api.openai_client import OpenAIClient
        from mirix.schemas.enums import MessageRole
        from mirix.schemas.llm_config import LLMConfig
        from mirix.schemas.message import Message
        from mirix.schemas.mirix_message_content import TextContent

        config = LLMConfig(
            model="fake-model",
            model_endpoint_type="openai",
            model_endpoint="http://127.0.0.1:9/v1",
            context_window=32_000,
            api_key="fake-key",
            parallel_tool_calls=parallel,
        )
        messages = [
            Message(role=MessageRole.system, content=[TextContent(text="s")]),
            Message(role=MessageRole.user, content=[TextContent(text="u")]),
        ]
        return await OpenAIClient(config).build_request_data(
            messages, config, tools=tools
        )

    async def test_it_is_sent_only_when_set_and_tools_are_offered(self):
        tools = [_tool("skill_list").json_schema]
        assert "parallel_tool_calls" not in await self._request(
            parallel=None, tools=tools
        )
        assert (await self._request(parallel=False, tools=tools))[
            "parallel_tool_calls"
        ] is False
        assert (await self._request(parallel=True, tools=tools))[
            "parallel_tool_calls"
        ] is True
        assert "parallel_tool_calls" not in await self._request(
            parallel=False, tools=None
        )
