"""Side-by-side comparison rendering for the ``--arm both`` invocation.

Pure functions only — given two :class:`RunResult` objects (one per arm) plus
a few metadata fields, produce the ``reports.md`` markdown shown to the user.

Kept as its own module so the rendering logic is unit-testable without
spinning up subprocesses (see ``tests/test_comparison.py``).

The token columns surface ``summary.tokens.agent.total_input`` and
``summary.tokens.agent.output`` from the vendored bench's ``report.json``.
When the upstream API doesn't report usage (e.g. OpenRouter for
``openai/gpt-5.2`` as of 2026-05-28), those values come through as ``0``;
we render them as ``"n/a"`` and skip the input-token delta line rather
than printing a misleading ``+0.0%``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .runner import RunResult


# ---------------------------------------------------------------------------
# Per-arm row extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmRow:
    """Flat per-arm row used by the markdown renderer.

    Separating extraction from rendering keeps the table format trivial to
    eyeball-test and lets the failure-arm path reuse the same renderer.
    """

    arm: str
    status: str  # "ok" or "failed (rc=N)"
    rounds: Optional[int]  # total_questions
    correct: Optional[float]
    accuracy: Optional[float]  # 0..1
    agent_input_tokens: Optional[int]  # summary.tokens.agent.total_input
    agent_output_tokens: Optional[int]  # summary.tokens.agent.output


def _extract_agent_tokens(summary: dict) -> tuple[Optional[int], Optional[int]]:
    """Return ``(total_input, output)`` from ``summary.tokens.agent``.

    Returns ``(None, None)`` if the summary is missing entirely (e.g. when
    the bench crashed before writing ``report.json``).  Returns ``0`` rather
    than ``None`` when the bench reported a zero — the caller renders ``0``
    as ``"n/a"`` because real-world this means "upstream LLM didn't report
    usage", not "agent actually produced zero tokens".
    """
    tokens = (summary.get("tokens") or {}) if isinstance(summary, dict) else {}
    agent = tokens.get("agent") or {}
    if not isinstance(agent, dict):
        return None, None
    ti = agent.get("total_input")
    out = agent.get("output")
    ti_int = int(ti) if isinstance(ti, (int, float)) else None
    out_int = int(out) if isinstance(out, (int, float)) else None
    return ti_int, out_int


def row_from_result(result: RunResult) -> ArmRow:
    """Project a :class:`RunResult` into the flat shape the renderer wants."""
    summary = result.report_summary or {}
    rounds_val = summary.get("total_questions")
    rounds = int(rounds_val) if isinstance(rounds_val, (int, float)) else None
    correct_val = summary.get("correct")
    correct = float(correct_val) if isinstance(correct_val, (int, float)) else None
    accuracy = result.accuracy
    ai, ao = _extract_agent_tokens(summary)

    if result.exit_code == 0:
        status = "ok"
    else:
        status = f"failed (rc={result.exit_code})"

    return ArmRow(
        arm=result.arm,
        status=status,
        rounds=rounds,
        correct=correct,
        accuracy=accuracy,
        agent_input_tokens=ai,
        agent_output_tokens=ao,
    )


# ---------------------------------------------------------------------------
# Cell formatters
# ---------------------------------------------------------------------------


def _fmt_token(n: Optional[int]) -> str:
    """Format token counts.  ``None`` or ``0`` -> ``"n/a"`` (see module docstring)."""
    if n is None or n == 0:
        return "n/a"
    return f"{n:,}"


def _fmt_accuracy(acc: Optional[float], status: str) -> str:
    if status != "ok" or acc is None:
        return "N/A"
    return f"{acc * 100:.1f}%"


def _fmt_rounds(n: Optional[int]) -> str:
    return "N/A" if n is None else str(n)


def _fmt_correct(c: Optional[float]) -> str:
    if c is None:
        return "N/A"
    # The bench writes 'correct' as a float (sum of per-question scores in
    # 0..1).  Render as int when integral, else 1-decimal float.
    if float(c).is_integer():
        return str(int(c))
    return f"{c:.1f}"


# ---------------------------------------------------------------------------
# Delta line
# ---------------------------------------------------------------------------


def _delta_line(metaclaw: ArmRow, mirix: ArmRow) -> str:
    """Build the ``**Delta**: ...`` line.

    Returns the empty string when neither arm produced an accuracy number
    (both failed) — we don't fabricate a delta out of two N/As.
    """
    parts: list[str] = []

    if (
        metaclaw.status == "ok"
        and mirix.status == "ok"
        and metaclaw.accuracy is not None
        and mirix.accuracy is not None
    ):
        acc_delta_pp = (mirix.accuracy - metaclaw.accuracy) * 100.0
        sign = "+" if acc_delta_pp >= 0 else ""
        parts.append(f"{sign}{acc_delta_pp:.1f}pp accuracy")
    else:
        parts.append(
            "accuracy delta unavailable (one or both arms failed or missing report)"
        )

    # Input-token delta only when both arms reported nonzero usage.
    if (
        metaclaw.agent_input_tokens
        and mirix.agent_input_tokens
        and metaclaw.agent_input_tokens > 0
    ):
        pct = (
            (mirix.agent_input_tokens - metaclaw.agent_input_tokens)
            / metaclaw.agent_input_tokens
        ) * 100.0
        sign = "+" if pct >= 0 else ""
        parts.append(f"{sign}{pct:.1f}% input tokens")
    else:
        # OpenRouter gpt-5.2 doesn't report usage -> tokens are 0; skip
        # rather than print a misleading +0.0%.  See module docstring.
        parts.append("input-token delta n/a (upstream did not report usage)")

    return "**Delta**: mirix - metaclaw = " + ", ".join(parts)


# ---------------------------------------------------------------------------
# Public renderer
# ---------------------------------------------------------------------------


def render_reports_md(
    metaclaw_result: RunResult,
    mirix_result: RunResult,
    *,
    days: int,
    kept_tests: int,
    vendor_sha: str,
    generated_at: Optional[datetime] = None,
    metaclaw_subdir: str = "",
    mirix_subdir: str = "",
) -> str:
    """Render the combined comparison markdown.

    Args:
        metaclaw_result / mirix_result: the two arm outcomes; either may be a
            failed run.  The renderer never raises on a failed arm.
        days: ``--days N`` argument as the user passed it (``0`` displayed as
            ``"all"``).
        kept_tests: the actual number of dataset days kept after slicing.
        vendor_sha: the pinned upstream MetaClaw SHA, displayed for
            reproducibility.
        generated_at: timestamp to print; defaults to ``datetime.now(UTC)``.
            Parametrised so tests get a deterministic header.
        metaclaw_subdir / mirix_subdir: relative paths from the parent
            ``runs/both-<ts>/`` dir to the per-arm dirs.  Empty string means
            "skip the per-arm report links section".
    """
    if generated_at is None:
        generated_at = datetime.now(timezone.utc)

    m_row = row_from_result(metaclaw_result)
    x_row = row_from_result(mirix_result)

    days_str = "all" if days == 0 else str(days)
    ts_str = generated_at.strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: list[str] = []
    lines.append("# Comparison Report")
    lines.append("")
    lines.append(f"Generated: {ts_str}")
    lines.append(f"Days: {days_str} ({kept_tests} day(s) kept)")
    lines.append(f"Vendor: aiming-lab/MetaClaw @ {vendor_sha or '(unknown)'}")
    lines.append("")
    lines.append(
        "| Arm | Status | Rounds | Correct | Accuracy | Agent In Tokens | Agent Out Tokens |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for row in (m_row, x_row):
        lines.append(
            f"| {row.arm} "
            f"| {row.status} "
            f"| {_fmt_rounds(row.rounds)} "
            f"| {_fmt_correct(row.correct)} "
            f"| {_fmt_accuracy(row.accuracy, row.status)} "
            f"| {_fmt_token(row.agent_input_tokens)} "
            f"| {_fmt_token(row.agent_output_tokens)} |"
        )
    lines.append("")
    lines.append(_delta_line(m_row, x_row))
    lines.append("")

    if metaclaw_subdir or mirix_subdir:
        lines.append("## Per-arm reports")
        if metaclaw_subdir:
            lines.append(
                f"- [metaclaw arm report]({metaclaw_subdir}/bench_output/report.md) "
                f"(meta: [{metaclaw_subdir}/run.meta.json]({metaclaw_subdir}/run.meta.json))"
            )
        if mirix_subdir:
            lines.append(
                f"- [mirix arm report]({mirix_subdir}/bench_output/report.md) "
                f"(meta: [{mirix_subdir}/run.meta.json]({mirix_subdir}/run.meta.json))"
            )
        lines.append("")

    return "\n".join(lines)


__all__ = [
    "ArmRow",
    "render_reports_md",
    "row_from_result",
]
