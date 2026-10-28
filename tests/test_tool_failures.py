"""The structured summary of an agent step's failed tool calls (``mirix/agent/tool_failures.py``).

Offline and pure: no database, no model. The agent's own recording of each stage is
tested with the real ``Agent.step`` in ``test_skill_curator_run.py``.
"""

import json
from datetime import datetime, timedelta

import pytest

from mirix.agent.tool_failures import (
    MAX_DISTINCT_TOOL_FAILURES,
    MAX_TOOL_FAILURE_ERROR_CHARS,
    ToolFailureLog,
)
from mirix.schemas.usage import MirixUsageStatistics, ToolFailure


def _dbapi_error():
    from sqlalchemy.exc import DBAPIError

    return DBAPIError(
        "SELECT 1",
        {},
        Exception("can't subtract offset-naive and offset-aware datetimes"),
    )


def test_identical_failures_are_counted_once_with_their_first_error():
    log = ToolFailureLog()
    log.record("validation", "skill_create", error="name must not be empty")
    log.record("validation", "skill_create", error="name must not be empty either")
    log.record("arguments", "skill_create", exc=ValueError("not a JSON object"))
    log.record("execution", "skill_read", exc=KeyError("x" * 1000))
    assert log.total == 4 and log.omitted == 0
    assert log.as_list() == [
        {
            "tool": "skill_create",
            "stage": "arguments",
            "exception": "ValueError",
            "error": "ValueError: not a JSON object",
            "count": 1,
            "argument_error": True,
        },
        {
            "tool": "skill_create",
            "stage": "validation",
            "exception": None,
            "error": "name must not be empty",
            "count": 2,
            "argument_error": True,
        },
        {
            "tool": "skill_read",
            "stage": "execution",
            "exception": "KeyError",
            "error": ("KeyError: '" + "x" * 1000)[:MAX_TOOL_FAILURE_ERROR_CHARS],
            "count": 1,
            "argument_error": False,
        },
    ]
    assert all(len(e["error"]) <= MAX_TOOL_FAILURE_ERROR_CHARS for e in log.as_list())


def test_the_exception_class_names_a_database_error_inside_a_tool():
    log = ToolFailureLog()
    log.record("execution", "list_memory_within_timerange", exc=_dbapi_error())
    (entry,) = log.as_list()
    assert entry["exception"] == "DBAPIError"
    assert entry["error"].startswith("DBAPIError: ")
    assert "offset-naive and offset-aware" in entry["error"]


def test_the_log_is_bounded_and_counts_what_did_not_fit():
    log = ToolFailureLog()
    for i in range(MAX_DISTINCT_TOOL_FAILURES + 7):
        log.record("unknown_tool", f"tool_{i:03d}", error=f"No function named tool_{i}")
    log.record("unknown_tool", "tool_000", error="again")  # an entry that exists
    entries = log.as_list()
    assert len(entries) == MAX_DISTINCT_TOOL_FAILURES
    assert log.omitted == 7 and log.total == MAX_DISTINCT_TOOL_FAILURES + 8
    assert sum(e["count"] for e in entries) + log.omitted == log.total
    assert entries == sorted(entries, key=lambda e: e["tool"])  # deterministic
    with pytest.raises(ValueError):
        log.record("somewhere", "t")  # type: ignore[arg-type]


def _raised(function, *args, **kwargs) -> BaseException:
    """The exception ``function`` raises, with its traceback."""
    try:
        function(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - the exception is the result
        return exc
    raise AssertionError(f"{function.__name__} did not raise")


def _mirix_value_error():
    # A ValueError of MIRIX's own code while a tool runs (function_sets/base.py
    # raises it when the agent has no user): not the model's mistake.
    raise ValueError("Can not list memory within timerange. User is not set")


def _pydantic_validation_error() -> BaseException:
    from pydantic import BaseModel

    class Item(BaseModel):
        count: int

    return _raised(Item, count="many")


def test_whether_a_failure_is_the_models_mistake_is_decided_on_the_exception():
    """The harness matched exception class NAMES, so a subclass of
    ValueError (a date parser's ParserError) or a missing argument (TypeError)
    read as MIRIX's own failure. MIRIX has the exception object: it records
    the verdict."""
    from dateutil.parser import ParserError

    from mirix.agent.tool_failures import is_argument_error
    from mirix.errors import ToolArgumentError
    from mirix.utils import parse_datetime_arg

    for exc in (
        ParserError("Unknown string format: %s", "string"),
        json.JSONDecodeError("Expecting value", "", 0),
        ToolArgumentError("skill_edit(): missing a required argument: 'field'"),
        _pydantic_validation_error(),
        # MIRIX's own parser of a time the model passed (a plain ValueError).
        _raised(parse_datetime_arg, "next tuesday-ish"),
        _raised(parse_datetime_arg, "2025-13-45", "UTC", end_of_day=True),
    ):
        assert is_argument_error("execution", exc), exc
    for exc in (TypeError("unsupported operand"), KeyError("k"), _dbapi_error(), None):
        assert not is_argument_error("execution", exc), exc
    for stage in ("unknown_tool", "arguments", "validation", "error_result"):
        assert is_argument_error(stage, None)
    log = ToolFailureLog()
    log.record("execution", "t", exc=ParserError("bad %s", "date"))
    log.record("execution", "u", exc=TypeError("MIRIX bug"))
    assert [e["argument_error"] for e in log.as_list()] == [True, False]


def test_an_out_of_range_value_inside_an_argument_parser_is_the_models():
    """An OverflowError raised while MIRIX's argument parser ran on the
    model's value (a date past the datetime range) is the model's value, not MIRIX's
    failure; the same error elsewhere is MIRIX's."""
    from mirix.agent import tool_failures
    from mirix.agent.tool_failures import is_argument_error

    def parse_datetime_arg(value):  # an argument parser by its (module, name)
        return datetime.max + timedelta(days=int(value))

    parse_datetime_arg.__module__ = "mirix.utils"
    namespace = {
        "__name__": "mirix.utils",
        "datetime": datetime,
        "timedelta": timedelta,
    }
    parser = type(parse_datetime_arg)(
        parse_datetime_arg.__code__, namespace, "parse_datetime_arg"
    )
    assert ("mirix.utils", "parse_datetime_arg") in tool_failures.ARGUMENT_PARSERS
    assert is_argument_error("execution", _raised(parser, "1"))
    assert not is_argument_error(
        "execution", _raised(lambda: datetime.max + timedelta(days=1))
    )


def test_a_plain_value_error_of_mirix_s_own_code_is_not_the_models():
    """Every ValueError at the execution stage counted as the
    model's mistake, so MIRIX's own ValueErrors (e.g. "Can not list memory
    within timerange. User is not set", function_sets/base.py) never reached
    the tool-failure warnings. A plain ValueError, or a subclass that is not an
    argument error, is MIRIX's; so is an id MIRIX looked up that names
    nothing (NoResultFound: the skill tools answer a model's unknown id
    themselves). A ValueError is a parser's verdict only when it was raised
    while one of MIRIX's argument parsers ran."""
    from mirix.agent.tool_failures import is_argument_error
    from mirix.orm.errors import NoResultFound, UniqueConstraintViolationError

    assert not is_argument_error("execution", ValueError("x"))
    assert not is_argument_error("execution", _raised(_mirix_value_error))
    assert not is_argument_error("execution", UniqueConstraintViolationError("dup"))
    assert not is_argument_error("execution", NoResultFound("no agent a-1"))
    # Not raised by one of MIRIX's argument parsers: int() is no verdict.
    assert not is_argument_error("execution", _raised(int, "not a number"))
    log = ToolFailureLog()
    log.record(
        "execution", "list_memory_within_timerange", exc=_raised(_mirix_value_error)
    )
    [entry] = log.as_list()
    assert entry["exception"] == "ValueError" and entry["argument_error"] is False
    # Every other stage is still the model's, whatever the exception.
    assert is_argument_error("arguments", ValueError("not a JSON object"))


def test_logs_of_one_step_merge():
    first, second = ToolFailureLog(max_distinct=2), ToolFailureLog()
    first.record("validation", "a", error="bad")
    second.record("validation", "a", error="bad again")
    second.record("execution", "b", exc=RuntimeError("boom"))
    second.record("execution", "c", exc=RuntimeError("boom"))
    first.merge(second)
    assert [(e["tool"], e["count"]) for e in first.as_list()] == [("a", 2), ("b", 1)]
    assert first.total == 4 and first.omitted == 1


def test_usage_statistics_carry_the_failures_and_round_trip():
    usage = MirixUsageStatistics()
    assert usage.tool_failures == [] and usage.tool_failures_omitted == 0
    log = ToolFailureLog()
    log.record("execution", "list_memory_within_timerange", exc=_dbapi_error())
    usage = MirixUsageStatistics(
        failed_tool_calls=1, tool_failures=log.as_list(), tool_failures_omitted=0
    )
    assert isinstance(usage.tool_failures[0], ToolFailure)
    again = MirixUsageStatistics.model_validate_json(usage.model_dump_json())
    assert again == usage
    assert again.model_dump()["tool_failures"] == log.as_list()
    # An older client's payload without the fields still parses.
    assert (
        MirixUsageStatistics.model_validate({"failed_tool_calls": 2}).tool_failures
        == []
    )
