"""Why an agent step's tool calls failed, as a bounded, structured summary.

Every failed tool call is answered to the model as a failed tool result and counted
(``MirixUsageStatistics.failed_tool_calls``). The count alone cannot tell the model's
own mistakes (an unknown tool, arguments that are not JSON or fail validation, a value
the tool rejects) from MIRIX's own errors or an infrastructure failure inside a tool (a
database error, say). :class:`ToolFailureLog` keeps, per ``(tool, stage, exception
class)``, how often it happened and the first error text, so whoever reads the usage
(the skill curator's per-batch metrics) can tell them apart.

Stages (:data:`ToolFailureStage`), in the order an agent step meets them:

* ``unknown_tool``: the model named a tool the agent does not have;
* ``arguments``: its arguments are not a JSON object;
* ``validation``: they failed the tool's validator (``tool_validators``);
* ``execution``: the tool raised while it ran, or MIRIX raised around its run
  (preparing it, checking its result); ``exception`` names the class;
* ``error_result``: the tool returned an error text without raising.

Each entry also says whether the failure is the model's mistake in its call
(``argument_error``, :func:`is_argument_error`): every stage but ``execution`` is;
an ``execution`` failure only when its exception is an argument error — arguments
that do not fit the tool's signature (:class:`mirix.errors.ToolArgumentError`),
a payload its model refuses (pydantic's ``ValidationError``), JSON text that does
not parse, or a value one of MIRIX's argument parsers (:data:`ARGUMENT_PARSERS`)
or a date parser cannot read. It is decided on the exception object, so a subclass
counts; a plain ``ValueError`` (or ``TypeError``, ``NoResultFound``) raised by
MIRIX's own code while the tool ran does not, so MIRIX's own errors are never
hidden as the model's.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional, Tuple

ToolFailureStage = Literal[
    "unknown_tool", "arguments", "validation", "execution", "error_result"
]
TOOL_FAILURE_STAGES: Tuple[str, ...] = (
    "unknown_tool",
    "arguments",
    "validation",
    "execution",
    "error_result",
)
#: Distinct ``(tool, stage, exception)`` entries one step keeps; later new ones are only
#: counted (``omitted``), so a runaway step cannot grow its usage record without bound.
MAX_DISTINCT_TOOL_FAILURES = 50
#: The first error text kept per entry, in characters.
MAX_TOOL_FAILURE_ERROR_CHARS = 300


#: Stages that are always the model's mistake in its call.
MODEL_STAGES = frozenset({"unknown_tool", "arguments", "validation", "error_result"})


#: MIRIX's parsers of an argument value a model passed, as ``(module, function)``:
#: a ``ValueError`` raised while one of them ran (by the parser or by what it called
#: on the value, e.g. ``datetime.strptime``: a frame of the traceback is the parser's)
#: is its verdict that the model's value cannot be read, and an ``OverflowError``
#: that the value is outside the range it can represent (a date past year 9999).
#: Every caller passes it a value from the model's call.
ARGUMENT_PARSERS = frozenset({("mirix.utils", "parse_datetime_arg")})


def _argument_error_classes() -> Tuple[type, ...]:
    """The exception classes that are argument errors wherever they are raised."""
    from pydantic import ValidationError

    from mirix.errors import ToolArgumentError

    classes: List[type] = [ToolArgumentError, ValidationError, json.JSONDecodeError]
    try:  # a date parser's "cannot read this string" (python-dateutil)
        from dateutil.parser import ParserError
    except ImportError:  # pragma: no cover - dateutil is not a MIRIX requirement
        pass
    else:
        classes.append(ParserError)
    return tuple(classes)


def _raised_in_argument_parser(exc: BaseException) -> bool:
    """Whether ``exc`` was raised while one of :data:`ARGUMENT_PARSERS` ran."""
    tb = exc.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        if (frame.f_globals.get("__name__"), frame.f_code.co_name) in ARGUMENT_PARSERS:
            return True
        tb = tb.tb_next
    return False


def is_argument_error(stage: str, exc: Optional[BaseException]) -> bool:
    """Whether a failed call is the model's mistake in its call: any stage but
    ``execution``; there, only an argument error — :class:`mirix.errors.ToolArgumentError`
    (arguments that do not fit the tool's signature), pydantic's ``ValidationError``,
    ``json.JSONDecodeError``, a date parser's ``ParserError``, or a ``ValueError`` or
    ``OverflowError`` raised while one of :data:`ARGUMENT_PARSERS` ran. Any other
    exception (a plain ``ValueError`` or ``OverflowError``, a ``TypeError``,
    ``NoResultFound``, a database error) came from MIRIX's own code or its
    environment."""
    if stage in MODEL_STAGES:
        return True
    if exc is None:
        return False
    if isinstance(exc, _argument_error_classes()):
        return True
    return isinstance(exc, (ValueError, OverflowError)) and _raised_in_argument_parser(
        exc
    )


def exception_text(exc: BaseException) -> str:
    """``"<Class>: <message>"``, cut to :data:`MAX_TOOL_FAILURE_ERROR_CHARS`."""
    return f"{type(exc).__name__}: {exc}"[:MAX_TOOL_FAILURE_ERROR_CHARS]


class ToolFailureLog:
    """The failed tool calls of one agent step, aggregated by ``(tool, stage,
    exception class)`` with a count and the first error text.

    ``total`` is every recorded failure (it equals the step's ``failed_tool_calls``);
    ``omitted`` those that did not fit :data:`MAX_DISTINCT_TOOL_FAILURES`
    (``sum(count) + omitted == total``)."""

    def __init__(self, max_distinct: int = MAX_DISTINCT_TOOL_FAILURES) -> None:
        self.max_distinct = max_distinct
        self._entries: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self.total = 0
        self.omitted = 0

    def record(
        self,
        stage: ToolFailureStage,
        tool: Optional[str],
        *,
        exc: Optional[BaseException] = None,
        error: Optional[str] = None,
    ) -> None:
        """One failed tool call: its ``stage``, the ``tool`` the model named, and the
        exception it raised (``exc``) or the error text it was answered with
        (``error``; ``exc``'s own text when only ``exc`` is given)."""
        if stage not in TOOL_FAILURE_STAGES:
            raise ValueError(f"unknown tool failure stage {stage!r}")
        self.total += 1
        name = type(exc).__name__ if exc is not None else None
        key = (str(tool or ""), stage, name or "")
        entry = self._entries.get(key)
        if entry is not None:
            entry["count"] += 1
            return
        if len(self._entries) >= self.max_distinct:
            self.omitted += 1
            return
        text = error if error is not None else (exception_text(exc) if exc else None)
        self._entries[key] = {
            "tool": str(tool or ""),
            "stage": stage,
            "exception": name,
            "error": text[:MAX_TOOL_FAILURE_ERROR_CHARS] if text is not None else None,
            "count": 1,
            "argument_error": is_argument_error(stage, exc),
        }

    def merge(self, other: "ToolFailureLog") -> None:
        """Add ``other``'s failures (a later part of the same step) to this log."""
        for entry in other.as_list():
            key = (entry["tool"], entry["stage"], entry["exception"] or "")
            mine = self._entries.get(key)
            if mine is not None:
                mine["count"] += entry["count"]
            elif len(self._entries) < self.max_distinct:
                self._entries[key] = dict(entry)
            else:
                self.omitted += entry["count"]
        self.total += other.total
        self.omitted += other.omitted

    def as_list(self) -> List[Dict[str, Any]]:
        """The entries, sorted by tool, stage order and exception (deterministic)."""
        return [
            dict(self._entries[key])
            for key in sorted(
                self._entries,
                key=lambda k: (k[0], TOOL_FAILURE_STAGES.index(k[1]), k[2]),
            )
        ]
