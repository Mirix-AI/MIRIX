"""Truth table for ``derive_write_kind`` (R2).

The write kind is fully determined by the queue message: conversation payload
(unified ``messages`` field 27, or legacy ``input_messages`` field 3) vs
``direct_writes`` (field 25). Both present → ``mixed`` — NEVER one of the pure
kinds (R2 AC2), so eval tooling can select pure-extraction traces with one
exact-match filter.

The no-payload combination is unreachable in practice — ECMS validates that at
least one of messages/direct_writes is present before enqueueing
(``save.py:288-297``) and the consumer-side normalization rejects malformed
messages — but the helper is total: it maps that combination to ``extraction``
(the historical default for a message with no direct writes).
"""

import pytest

from mirix.queue.message_pb2 import QueueMessage
from mirix.queue.worker import derive_write_kind


def _msg(with_messages=False, with_input_messages=False, with_direct_writes=False):
    m = QueueMessage()
    m.agent_id = "agent-1"
    if with_messages:
        pm = m.messages.add()
        pm.text_content = "hello"
    if with_input_messages:
        pm = m.input_messages.add()
        pm.text_content = "hello (legacy packed)"
    if with_direct_writes:
        w = m.direct_writes.add()
        w.memory_type = "episodic"
        w.payload_json = "{}"
    return m


@pytest.mark.parametrize(
    "with_messages,with_input_messages,with_direct_writes,expected",
    [
        # Pure conversation payload → extraction (either wire shape).
        (True, False, False, "extraction"),
        (False, True, False, "extraction"),
        (True, True, False, "extraction"),
        # Pure direct writes → direct.
        (False, False, True, "direct"),
        # Both → mixed, never a pure kind (R2 AC2).
        (True, False, True, "mixed"),
        (False, True, True, "mixed"),
        (True, True, True, "mixed"),
        # No payload: unreachable upstream (ECMS validation); helper is total
        # and keeps the historical default.
        (False, False, False, "extraction"),
    ],
)
def test_write_kind_truth_table(with_messages, with_input_messages, with_direct_writes, expected):
    message = _msg(with_messages, with_input_messages, with_direct_writes)
    assert derive_write_kind(message) == expected
