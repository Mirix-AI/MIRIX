"""Tests for :mod:`evals.metaclaw.comparison`.

All pure-function tests — no subprocesses, no I/O.  Build :class:`RunResult`
objects directly and assert against the rendered markdown.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


from evals.metaclaw.comparison import (
    ArmRow,
    _delta_line,
    _extract_agent_tokens,
    _fmt_accuracy,
    _fmt_correct,
    _fmt_rounds,
    _fmt_token,
    render_reports_md,
    row_from_result,
)
from evals.metaclaw.runner import RunResult


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _ok_result(
    arm: str,
    accuracy: float,
    *,
    total_questions: int = 10,
    correct: float = 4.0,
    agent_input: int = 0,
    agent_output: int = 0,
) -> RunResult:
    return RunResult(
        arm=arm,
        exit_code=0,
        output_dir=Path("/tmp/fake"),
        accuracy=accuracy,
        total_tokens=agent_input + agent_output,
        report_summary={
            "total_questions": total_questions,
            "correct": correct,
            "accuracy": accuracy,
            "tokens": {
                "agent": {
                    "input": agent_input,
                    "output": agent_output,
                    "cache_read": 0,
                    "total_input": agent_input,
                },
                "compaction": {
                    "input": 0,
                    "output": 0,
                    "cache_read": 0,
                    "total_input": 0,
                },
            },
        },
    )


def _failed_result(arm: str, rc: int = 2) -> RunResult:
    return RunResult(
        arm=arm,
        exit_code=rc,
        output_dir=Path("/tmp/fake"),
        accuracy=None,
        total_tokens=None,
        report_summary={"error": "boom"},
    )


_DET_TS = datetime(2026, 5, 28, 14, 23, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Cell formatters
# ---------------------------------------------------------------------------


class TestFormatters:
    def test_fmt_token_zero_is_na(self) -> None:
        # OpenRouter gpt-5.2 doesn't report usage -> 0 surfaces as n/a
        # rather than a misleading literal 0.
        assert _fmt_token(0) == "n/a"
        assert _fmt_token(None) == "n/a"

    def test_fmt_token_nonzero_is_grouped(self) -> None:
        assert _fmt_token(1234567) == "1,234,567"
        assert _fmt_token(42) == "42"

    def test_fmt_accuracy_ok(self) -> None:
        assert _fmt_accuracy(0.611, "ok") == "61.1%"
        assert _fmt_accuracy(0.0, "ok") == "0.0%"
        assert _fmt_accuracy(1.0, "ok") == "100.0%"

    def test_fmt_accuracy_failed_or_missing(self) -> None:
        assert _fmt_accuracy(0.5, "failed (rc=2)") == "N/A"
        assert _fmt_accuracy(None, "ok") == "N/A"

    def test_fmt_rounds(self) -> None:
        assert _fmt_rounds(36) == "36"
        assert _fmt_rounds(None) == "N/A"

    def test_fmt_correct_integral_renders_as_int(self) -> None:
        assert _fmt_correct(4.0) == "4"
        assert _fmt_correct(0.0) == "0"

    def test_fmt_correct_fractional(self) -> None:
        assert _fmt_correct(3.5) == "3.5"

    def test_fmt_correct_none(self) -> None:
        assert _fmt_correct(None) == "N/A"


# ---------------------------------------------------------------------------
# row_from_result + token extraction
# ---------------------------------------------------------------------------


class TestRowExtraction:
    def test_row_from_ok_result(self) -> None:
        r = _ok_result(
            "mirix",
            0.69,
            total_questions=36,
            correct=25,
            agent_input=1_250_000,
            agent_output=92_000,
        )
        row = row_from_result(r)
        assert row == ArmRow(
            arm="mirix",
            status="ok",
            rounds=36,
            correct=25.0,
            accuracy=0.69,
            agent_input_tokens=1_250_000,
            agent_output_tokens=92_000,
        )

    def test_row_from_failed_result(self) -> None:
        row = row_from_result(_failed_result("mirix", rc=2))
        assert row.status == "failed (rc=2)"
        assert row.accuracy is None
        assert row.rounds is None
        assert row.correct is None
        assert row.agent_input_tokens is None
        assert row.agent_output_tokens is None

    def test_extract_agent_tokens_missing_summary(self) -> None:
        assert _extract_agent_tokens({}) == (None, None)

    def test_extract_agent_tokens_zero(self) -> None:
        ti, ao = _extract_agent_tokens(
            {"tokens": {"agent": {"total_input": 0, "output": 0}}}
        )
        assert ti == 0
        assert ao == 0


# ---------------------------------------------------------------------------
# Delta line
# ---------------------------------------------------------------------------


class TestDeltaLine:
    def test_positive_delta(self) -> None:
        m = row_from_result(_ok_result("metaclaw", 0.611, agent_input=1_000_000))
        x = row_from_result(_ok_result("mirix", 0.694, agent_input=1_010_000))
        line = _delta_line(m, x)
        assert line.startswith("**Delta**: mirix - metaclaw = ")
        assert "+8.3pp accuracy" in line
        assert "+1.0% input tokens" in line

    def test_negative_accuracy_delta(self) -> None:
        m = row_from_result(_ok_result("metaclaw", 0.700, agent_input=1_000_000))
        x = row_from_result(_ok_result("mirix", 0.500, agent_input=900_000))
        line = _delta_line(m, x)
        assert "-20.0pp accuracy" in line
        assert "-10.0% input tokens" in line

    def test_token_zero_skips_token_delta(self) -> None:
        m = row_from_result(_ok_result("metaclaw", 0.5, agent_input=0))
        x = row_from_result(_ok_result("mirix", 0.6, agent_input=0))
        line = _delta_line(m, x)
        assert "+10.0pp accuracy" in line
        assert "n/a" in line
        # Must not print a fake percentage when upstream didn't report usage.
        assert "0.0% input tokens" not in line

    def test_one_arm_failed_no_acc_delta(self) -> None:
        m = row_from_result(_ok_result("metaclaw", 0.6, agent_input=100))
        x = row_from_result(_failed_result("mirix", rc=2))
        line = _delta_line(m, x)
        assert "accuracy delta unavailable" in line


# ---------------------------------------------------------------------------
# render_reports_md
# ---------------------------------------------------------------------------


class TestRenderReportsMd:
    def test_both_ok_full_table(self) -> None:
        m = _ok_result(
            "metaclaw",
            0.611,
            total_questions=36,
            correct=22,
            agent_input=1_234_567,
            agent_output=89_012,
        )
        x = _ok_result(
            "mirix",
            0.694,
            total_questions=36,
            correct=25,
            agent_input=1_250_000,
            agent_output=92_000,
        )
        md = render_reports_md(
            m,
            x,
            days=3,
            kept_tests=3,
            vendor_sha="deadbeef",
            generated_at=_DET_TS,
            metaclaw_subdir="metaclaw-20260528T142301Z",
            mirix_subdir="mirix-20260528T142301Z",
        )
        assert "# Comparison Report" in md
        assert "Generated: 2026-05-28 14:23:01 UTC" in md
        assert "Vendor: aiming-lab/MetaClaw @ deadbeef" in md
        # Header row + separator row
        assert (
            "| Arm | Status | Rounds | Correct | Accuracy | Agent In Tokens | Agent Out Tokens |"
            in md
        )
        assert "|---|---|---|---|---|---|---|" in md
        # Data rows
        assert "| metaclaw | ok | 36 | 22 | 61.1% | 1,234,567 | 89,012 |" in md
        assert "| mirix | ok | 36 | 25 | 69.4% | 1,250,000 | 92,000 |" in md
        # Delta line
        assert "**Delta**: mirix - metaclaw = +8.3pp accuracy, +1.3% input tokens" in md
        # Per-arm report links
        assert (
            "[metaclaw arm report](metaclaw-20260528T142301Z/bench_output/report.md)"
            in md
        )
        assert "[mirix arm report](mirix-20260528T142301Z/bench_output/report.md)" in md

    def test_zero_tokens_render_as_na(self) -> None:
        # Real OpenRouter gpt-5.2 case: no usage reported.
        m = _ok_result("metaclaw", 0.5, total_questions=10, correct=5, agent_input=0)
        x = _ok_result("mirix", 0.4, total_questions=10, correct=4, agent_input=0)
        md = render_reports_md(
            m, x, days=1, kept_tests=1, vendor_sha="abc", generated_at=_DET_TS
        )
        # Two n/a token cells per row, twice = 4 n/a tokens in the table area.
        # We don't pin the count but assert the rows are right.
        assert "| metaclaw | ok | 10 | 5 | 50.0% | n/a | n/a |" in md
        assert "| mirix | ok | 10 | 4 | 40.0% | n/a | n/a |" in md
        assert "-10.0pp accuracy" in md
        assert "input-token delta n/a" in md

    def test_metaclaw_arm_failed(self) -> None:
        m = _failed_result("metaclaw", rc=137)
        x = _ok_result(
            "mirix", 0.55, total_questions=12, correct=6, agent_input=500_000
        )
        md = render_reports_md(
            m, x, days=2, kept_tests=2, vendor_sha="sha", generated_at=_DET_TS
        )
        assert "| metaclaw | failed (rc=137) | N/A | N/A | N/A | n/a | n/a |" in md
        assert "| mirix | ok | 12 | 6 | 55.0% | 500,000 | n/a |" in md
        assert "accuracy delta unavailable" in md

    def test_mirix_arm_failed(self) -> None:
        m = _ok_result("metaclaw", 0.6, total_questions=12, correct=7)
        x = _failed_result("mirix", rc=2)
        md = render_reports_md(
            m, x, days=2, kept_tests=2, vendor_sha="sha", generated_at=_DET_TS
        )
        assert "| metaclaw | ok | 12 | 7 | 60.0% |" in md
        assert "| mirix | failed (rc=2) | N/A | N/A | N/A | n/a | n/a |" in md

    def test_both_arms_failed_still_renders(self) -> None:
        m = _failed_result("metaclaw", rc=1)
        x = _failed_result("mirix", rc=2)
        md = render_reports_md(
            m, x, days=1, kept_tests=1, vendor_sha="sha", generated_at=_DET_TS
        )
        # Stub must still contain the table header and both rows.
        assert "# Comparison Report" in md
        assert "| metaclaw | failed (rc=1) |" in md
        assert "| mirix | failed (rc=2) |" in md
        assert "accuracy delta unavailable" in md

    def test_days_zero_displays_as_all(self) -> None:
        m = _ok_result("metaclaw", 0.5)
        x = _ok_result("mirix", 0.5)
        md = render_reports_md(
            m, x, days=0, kept_tests=30, vendor_sha="sha", generated_at=_DET_TS
        )
        assert "Days: all (30 day(s) kept)" in md

    def test_no_subdirs_skips_links_section(self) -> None:
        m = _ok_result("metaclaw", 0.5)
        x = _ok_result("mirix", 0.5)
        md = render_reports_md(
            m,
            x,
            days=1,
            kept_tests=1,
            vendor_sha="sha",
            generated_at=_DET_TS,
        )
        assert "## Per-arm reports" not in md
