"""The JSON-extraction helpers of LLM replies (``mirix/helpers/json_parsing.py``).

Pure and offline. The balanced-span scan behind every "object/array embedded in
prose" fallback must find exactly the spans the original per-start scan found
(each opening bracket, its shortest closing substring, brackets inside JSON
strings ignored), in linear time: the old scan restarted at every bracket and
read to the end of the text, so a reply with many unclosed braces took seconds
on the event loop (0.16 s at 70 KB, 1.3 s at 209 KB).
"""

import json
import random
import time

from mirix.helpers.json_parsing import (
    SPAN_SCAN_FACTOR,
    _balanced_arrays,
    _balanced_objects,
    parse_json_array_or_none,
    parse_json_object_with_key,
)


def _reference_spans(text, open_ch, close_ch):
    """The original scan: from every ``open_ch``, read until it closes."""
    out = []
    for i, first in enumerate(text):
        if first != open_ch:
            continue
        depth, in_str, escaped = 0, False, False
        for j in range(i, len(text)):
            ch = text[j]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    out.append(text[i : j + 1])
                    break
    return out


def test_the_spans_are_those_of_the_per_start_scan():
    rng = random.Random(20260930)
    alphabet = '{}[]"\\ab:,'
    for _ in range(5000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
        assert list(_balanced_objects(text)) == _reference_spans(text, "{", "}"), text
        assert list(_balanced_arrays(text)) == _reference_spans(text, "[", "]"), text


def test_nested_and_string_brackets():
    text = 'x {"a": {"b": "}{"}, "c": [1, {"d": 2}]} y {broken'
    assert list(_balanced_objects(text)) == [
        '{"a": {"b": "}{"}, "c": [1, {"d": 2}]}',
        '{"b": "}{"}',
        '{"d": 2}',
    ]
    assert list(_balanced_arrays(text)) == ['[1, {"d": 2}]']


def test_many_unclosed_braces_are_scanned_in_linear_time():
    lesson = {
        "experience_type": "worth_avoiding",
        "title": "t" * 50,
        "content": "c" * 400,
        "evidence": {"quote": "q" * 80, "signal_type": "tool_error"},
    }
    body = '{"outcome": "failed", "experiences": [' + ",".join(
        '{"a": ' + json.dumps(lesson) for _ in range(1000)
    )  # ~700 KB, every lesson wrapped in a brace that never closes
    start = time.perf_counter()
    assert parse_json_object_with_key(body, "experiences") is None
    items = parse_json_array_or_none(body)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, elapsed
    assert items == [lesson]  # the first balanced, parseable object
    assert SPAN_SCAN_FACTOR >= 2
