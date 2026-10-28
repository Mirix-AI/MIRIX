"""Which tool calls of one model turn run, and in what order.

A model may answer one turn with several tool calls. :func:`plan_tool_calls`
decides, before anything runs, which of them the agent executes. A
**terminal** call (:data:`TERMINAL_TOOLS`) ends the agent's chain; there are
two kinds:

* A **finish** (:data:`FINISH_TOOLS`: ``finish_memory_update``) ends the
  work: it runs after every other call of the turn, whatever its position in
  the answer, and once (an earlier finish is answered "not run").
* A **reply** (:data:`REPLY_TOOLS`: ``send_message``) is content for the
  user, so it is never dropped for a later one: every distinct reply runs, in
  the answer's order (an exact repeat, the same arguments, runs once). A reply
  runs after the turn's other calls when it comes after all of them in the
  answer; a reply that comes before another call of the turn is deferred: it
  is answered "not run" and the chain goes on, so the model reads the results
  of the calls it made after it (a search, a lookup) before it replies. A
  reply made in the same turn as a search can never include its result. The
  replies of a turn go out together: when one is deferred, every reply of the
  turn is (the model sends its whole reply in a later turn, never half now
  and a second reply later).

Only a tool the agent can call this step is terminal (``available``: its
tools after the step's ``tool_allowlist``). A call to any other name is an
ordinary call whatever its name (a memory sub-agent that hallucinates
``send_message``): the agent answers it "not available" in this turn, and
it never defers a finish or is asked to be made alone. Having no result a
reply could use, it never makes a reply wait; and a terminal name the agent
cannot call takes no slot of a per-turn cap (below).

With ``max_calls`` (a step that takes a bounded number of calls per turn,
e.g. the skill curator's one call per turn, the serial tool-call contract of
``parallel_tool_calls=false``): only the first ``max_calls`` non-terminal
calls run — a call to a terminal name the agent cannot call is not counted
among them (a hallucinated ``send_message`` is no step a later call builds
on), a call to any other name it cannot call is — (a later call to a tool
the agent cannot call is answered "not available" at once: it runs nothing,
so it is never asked for again), and
terminal calls run only when the turn has no other call, so
the model always reads every result before it ends the chain. A model that
is answered "call it alone" and pairs the terminal call again in its next
turn (``terminal_after_others``, after :data:`TERMINAL_ALONE_REMINDERS` such
turns in a row) is not held until the step's cap: the terminal calls then run
after the turn's other calls, provided they all fit the cap (a turn never
ends with one of its calls left unrun), and only when those calls went
through (``ToolCallPlan.guarded``: the agent holds the terminal calls when a
call before them was rejected, refused or failed, so the model reads that
answer first).

Every call that does not run is answered with a "Not run: …" tool result, so
each ``tool_call_id`` of the answer still gets its tool message (the agent
emits them in the answer's original order). Pure CPU: no I/O.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Collection, Dict, List, Optional, Sequence, Tuple

#: Tools that end the agent's chain (see the module docstring).
TERMINAL_TOOLS = ("send_message", "finish_memory_update")
#: Terminal tools that end the work: they run last and at most once per turn.
FINISH_TOOLS = ("finish_memory_update",)
#: Terminal tools that deliver a reply: each distinct call runs, in order.
REPLY_TOOLS = ("send_message",)
#: Turns in a row a terminal call is answered "call it alone" before a turn
#: that pairs it again runs it after its other calls: one reminder turns a
#: habit into an explicit instruction, and a model that still pairs the call
#: never spins until the step cap (the step would end unfinished).
TERMINAL_ALONE_REMINDERS = 1


@dataclass(frozen=True)
class ToolCallPlan:
    """``order``: the indices of the calls to run, in run order. ``skipped``:
    index -> the "Not run: …" text a call that does not run is answered with.
    ``deferred``: True when a call was left for a later turn (any skip other
    than a repeated call): the chain must go on so the model can make it
    again. ``terminal_deferred``: True when a terminal call was answered "call
    it alone" (the agent counts such turns in a row). ``guarded``: the
    indices of ``order`` that are terminal calls run after their turn's other
    calls on a repeat (``terminal_after_others``): the agent runs them only
    when every call before them went through (:func:`held_terminal_text`)."""

    order: List[int] = field(default_factory=list)
    skipped: Dict[int, str] = field(default_factory=dict)
    deferred: bool = False
    terminal_deferred: bool = False
    guarded: Tuple[int, ...] = ()


def _per_turn(max_calls: int) -> str:
    return (
        "one tool call per turn"
        if max_calls == 1
        else f"at most {max_calls} tool calls per turn"
    )


def _window_text(
    names: Sequence[str], window: Sequence[int], can_call: Sequence[bool]
) -> str:
    """What the calls in a capped turn's window did: the ones the agent could
    make ran; the others were answered "not available" (never "ran")."""
    ran = list(dict.fromkeys(names[i] for i in window if can_call[i]))
    lacking = list(dict.fromkeys(names[i] for i in window if not can_call[i]))
    parts = []
    if ran:
        parts.append(f"{', '.join(ran)} ran")
    if lacking:
        verb = "is" if len(lacking) == 1 else "are"
        parts.append(f"{', '.join(lacking)} {verb} not available")
    return "; ".join(parts)


def _argument_key(raw: Optional[str]) -> str:
    """A call's arguments for comparison: canonical JSON when they parse
    (key order and spacing do not matter), else the stripped text."""
    text = "" if raw is None else str(raw)
    try:
        return json.dumps(json.loads(text), sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return text.strip()


def held_terminal_text(name: str) -> str:
    """The answer of a guarded terminal call the agent holds because a call
    before it in the same turn did not go through."""
    return (
        "Not run: another call of this turn did not go through (it was "
        f"rejected, refused or failed). Read its answer, then call {name}() "
        "alone."
    )


def _identical(name: str) -> str:
    return f"Not run: an identical {name} call of this turn ran; it ran once."


def _reply_waits_too(earlier: str, name: str) -> str:
    return (
        f"Not run: an earlier {earlier} of this turn waits for the results of "
        f"the calls made after it, so this {name} waits too (a turn's replies go "
        "out together). Read those results, then send your reply."
    )


def plan_tool_calls(
    names: Sequence[Optional[str]],
    *,
    max_calls: Optional[int] = None,
    terminal_after_others: bool = False,
    arguments: Optional[Sequence[Optional[str]]] = None,
    available: Optional[Collection[str]] = None,
) -> ToolCallPlan:
    """The run order of a turn's tool calls named ``names`` (in the answer's
    order); see the module docstring. ``max_calls`` None (the default) runs
    every call; otherwise it must be >= 1. ``terminal_after_others`` (only
    meaningful with ``max_calls``): the model already was told to make its
    terminal call alone, so it runs after the others when they all run.
    ``arguments``: each call's raw arguments (JSON text), which tell an exact
    repeat of a reply from a different one; without them every reply counts
    as distinct. ``available``: the names of the tools the agent can call
    this step; a call to any other name is an ordinary call (never terminal,
    never capped). None: every name counts as callable."""
    if max_calls is not None and max_calls < 1:
        raise ValueError("max_calls must be at least 1")
    names = [str(n or "") for n in names]
    known = None if available is None else frozenset(str(n) for n in available)
    can_call = [known is None or name in known for name in names]
    keys = [
        _argument_key(arguments[i])
        if arguments is not None and i < len(arguments)
        else f"#{i}"  # unknown arguments: never a repeat
        for i in range(len(names))
    ]
    skipped: Dict[int, str] = {}
    deferred = False

    # A terminal name the agent cannot call is an ordinary call.
    terminal_calls = [
        i for i, n in enumerate(names) if n in TERMINAL_TOOLS and can_call[i]
    ]
    is_terminal = set(terminal_calls)
    others = [i for i in range(len(names)) if i not in is_terminal]
    finishes = [i for i in terminal_calls if names[i] in FINISH_TOOLS]
    replies = [i for i in terminal_calls if names[i] in REPLY_TOOLS]

    # The cap takes the turn's first calls, whatever they call — but a reply
    # or a finish the agent cannot call takes no slot: it is no step a later
    # call builds on (a curator leading every turn with a hallucinated
    # send_message would never run its real call). A later call to a tool
    # the agent cannot call runs nothing: it is answered "not available" at
    # once, never asked for again ("Call it again").
    slotted = [i for i in others if names[i] not in TERMINAL_TOOLS]
    window = slotted if max_calls is None else slotted[:max_calls]
    capped_out = [i for i in slotted[len(window) :] if can_call[i]]
    for index in capped_out:
        skipped[index] = (
            f"Not run: this step takes {_per_turn(max_calls)} "
            f"({_window_text(names, window, can_call)}), so you read each result "
            f"before your next call. Call {names[index]} again if you still "
            "need it."
        )
        deferred = True
    run_others = [i for i in others if i not in skipped]

    # Replies: the ones that end the turn run (each distinct one once, in the
    # answer's order); without a cap, one that another call it can make
    # follows waits for a later turn, and then every reply of the turn waits
    # with it (with a cap, terminal calls wait as a group below). A call to a
    # tool the agent cannot call makes no reply wait: it is answered "not
    # available", a result the reply could not use.
    callable_others = [i for i in others if can_call[i]]
    last_other = callable_others[-1] if callable_others else -1
    if max_calls is None:
        ending = [i for i in replies if i > last_other]
        waiting = [i for i in replies if i < last_other]
    else:
        ending, waiting = list(replies), []
    # A waiting reply that an ending one repeats is sent once, with it.
    ending_keys = {(names[i], keys[i]) for i in ending}
    repeats = [i for i in waiting if (names[i], keys[i]) in ending_keys]
    waiting = [i for i in waiting if (names[i], keys[i]) not in ending_keys]
    run_replies: List[int] = []
    if waiting:
        for index in sorted(waiting + repeats):
            later = ", ".join(
                dict.fromkeys(names[i] for i in callable_others if i > index)
            )
            skipped[index] = (
                f"Not run: {names[index]} ends your turn, and this turn called "
                f"{later} after it. Read those results, then send your reply."
            )
        for index in ending:
            skipped[index] = _reply_waits_too(names[waiting[0]], names[index])
        deferred = True
    else:
        sent = set()
        for index in ending:
            key = (names[index], keys[index])
            if key in sent:
                skipped[index] = _identical(names[index])
            else:
                sent.add(key)
                run_replies.append(index)
        for index in repeats:
            skipped[index] = _identical(names[index])

    terminal = sorted(run_replies + finishes)
    run_terminal: List[int] = []
    terminal_deferred = False
    guarded = False
    if terminal:
        alone = max_calls is not None and bool(others)
        if alone and terminal_after_others and not capped_out:
            alone, guarded = False, True  # reminded already, every call runs
        if alone:
            # The model must read the other results before it ends the chain.
            for index in terminal:
                skipped[index] = (
                    f"Not run: call {names[index]}() alone, in a turn of its own, "
                    "after reading the results of your other calls."
                )
            deferred = terminal_deferred = True
        else:
            run_terminal = list(run_replies)
            if finishes and deferred:
                # A call of this turn waits for a later turn: the work is not
                # done, so the finish waits too (it would end the chain).
                for index in finishes:
                    skipped[index] = (
                        "Not run: a call of this turn waits for a later turn; "
                        f"make it, then call {names[index]}() alone."
                    )
            elif finishes:
                last = finishes[-1]
                run_terminal.append(last)
                for index in finishes[:-1]:
                    # The chain ends once: only the last finish runs.
                    skipped[index] = (
                        f"Not run: {names[index]} was called more than once in "
                        "this turn; it ran once."
                    )
    return ToolCallPlan(
        order=run_others + run_terminal,
        skipped=skipped,
        deferred=deferred,
        terminal_deferred=terminal_deferred,
        guarded=tuple(run_terminal) if guarded else (),
    )
