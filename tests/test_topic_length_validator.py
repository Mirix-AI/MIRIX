"""Unit tests for ``Agent._enforce_topic_length``.

The meta agent extracts conversation topics via LLM and forwards them
to memory search as a ``;``-joined string. The downstream search
backend (IPS Search) caps text-field values at 200 characters, so
each ``;``-delimited segment must stay under that limit;
``_enforce_topic_length`` truncates any oversized segment so the
search still runs (degraded but not crashed). These tests pin its
behavior end-to-end and on a few edge cases.
"""

from unittest.mock import MagicMock

from mirix.agent.agent import Agent


def _agent() -> Agent:
    """Bare-minimum Agent instance for testing pure helpers."""
    agent = Agent.__new__(Agent)
    agent.agent_state = MagicMock()
    agent.agent_state.name = "meta_memory_agent"
    return agent


class TestEnforceTopicLength:
    def test_none_passes_through(self):
        assert _agent()._enforce_topic_length(None) is None

    def test_empty_string_passes_through(self):
        # Empty input short-circuits at the top of the validator and
        # passes through unchanged — preserves the existing contract.
        assert _agent()._enforce_topic_length("") == ""

    def test_short_single_topic_unchanged(self):
        assert _agent()._enforce_topic_length("hello") == "hello"

    def test_multiple_short_topics_unchanged(self):
        assert _agent()._enforce_topic_length("alpha; beta ;gamma") == "alpha;beta;gamma"

    def test_oversized_single_topic_truncated_to_200(self):
        topic = "x" * 250
        out = _agent()._enforce_topic_length(topic)
        assert out is not None
        assert len(out) == 200

    def test_only_oversized_segment_in_multi_topic_string_truncated(self):
        # Two short topics + one oversize → oversize gets clipped, others
        # are preserved.
        short_a = "alpha"
        short_b = "beta"
        long_c = "c" * 250
        out = _agent()._enforce_topic_length(f"{short_a}; {long_c}; {short_b}")
        parts = out.split(";")
        assert parts[0] == "alpha"
        assert len(parts[1]) == 200
        assert parts[2] == "beta"

    def test_four_topic_string_under_per_topic_limit_passes_through(self):
        """A realistic four-topic ``key_words`` string where each
        individual topic is under 200 chars but the joined whole is
        over (215). Per-topic enforcement leaves the segments untouched
        and rejoins them — only the total length grows; the validator
        only cares about per-segment length."""
        topics = (
            "QuickBooks Online payroll tax workflow for restaurant clients; "
            "Reviewing payroll register for accurate tax filing; "
            "QuickBooks subscription renewal date and details; "
            "Switching restaurant clients to monthly payroll runs"
        )
        out = _agent()._enforce_topic_length(topics)
        parts = out.split(";")
        assert len(parts) == 4
        for p in parts:
            assert len(p) <= 200

    def test_empty_segments_dropped(self):
        # Extra ;; and trailing/leading separators should not produce
        # empty topics in the output.
        out = _agent()._enforce_topic_length("alpha; ; beta;;")
        assert out == "alpha;beta"
