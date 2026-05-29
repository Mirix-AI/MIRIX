"""Benchmark run command.

Runs the full evaluation pipeline for a test dataset:
  infer → scoring → report

For directory input (multiple all_tests.json), each test set is processed
independently and a combined ``reports.md`` is generated at the top level.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

from src.infer.infer_cmd import _find_all_tests_files, _run_one_all_tests
from src.infer.query_reader import get_default_query_reader
from src.report.report_cmd import run_report
from src.scoring.scoring_cmd import run_scoring
from src.utils import resolve_path


# ---------------------------------------------------------------------------
# Combined report helpers
# ---------------------------------------------------------------------------


def _generate_combined_reports(
    out_dir: Path,
    reports: list[tuple[str, dict]],
) -> None:
    """Generate reports.md summarising all test sets side-by-side.

    Args:
        out_dir: Directory for reports.md.
        reports: List of (name, report_dict) tuples.
    """
    lines = [
        "# Combined Benchmark Report",
        "",
        "| Test Set | Questions | Correct | Accuracy"
        " | Agent Total In | Agent Out | Comp Total In | Comp Out |",
        "|----------|-----------|---------|----------"
        "|----------------|-----------|---------------|----------|",
    ]

    for name, report in reports:
        summary = report.get("summary", {})
        total = summary.get("total_questions", 0)
        correct = summary.get("correct", 0)
        acc = summary.get("accuracy", 0.0)
        agent_tok = summary.get("tokens", {}).get("agent", {})
        comp_tok = summary.get("tokens", {}).get("compaction", {})
        a_ti = agent_tok.get("total_input", 0)
        a_out = agent_tok.get("output", 0)
        c_ti = comp_tok.get("total_input", 0)
        c_out = comp_tok.get("output", 0)
        lines.append(
            f"| {name} | {total} | {correct} | {acc:.1%}"
            f" | {a_ti:,} | {a_out:,} | {c_ti:,} | {c_out:,} |"
        )

    lines.append("")
    md = "\n".join(lines)

    reports_md_path = out_dir / "reports.md"
    reports_md_path.write_text(md, encoding="utf-8")
    print(f"\nWritten: {reports_md_path}")
    print("\n" + md)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_run(
    input_arg: str,
    output_arg: str,
    workers: int = 1,
    retry: int = 0,
    scene_per_train: int | None = None,
    memory: bool = False,
    memory_proxy_port: int = 30000,
    buffer_turns: bool = False,
) -> None:
    """Run infer → scoring → report pipeline.

    Args:
        input_arg: Path to all_tests.json or a directory containing them.
        output_arg: Output directory path.
        workers: Maximum number of concurrent tests (default: 1).
        retry: Number of retries per failed question (default: 0).
        scene_per_train: If set, trigger ``metaclaw train-step`` every N scenes.
        memory: If True, trigger batch memory ingestion after each test scene.
        memory_proxy_port: MetaClaw proxy port for memory / buffer_turn calls.
        buffer_turns: If True, use incremental ingestion via buffer_turn per round
                      and flush_session after each scene.
    """
    input_path = resolve_path(input_arg)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    all_tests_files = _find_all_tests_files(input_path)
    if not all_tests_files:
        raise FileNotFoundError(f"No all_tests.json found under: {input_path}")

    is_dir_input = input_path.is_dir()

    # Prepare base output directory.  Always create a fresh directory so that
    # re-running does not mix results from different runs.
    out_base = resolve_path(output_arg)
    if out_base.exists():
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_base = out_base / f"run_{run_id}"
    out_base.mkdir(parents=True, exist_ok=True)

    query_reader = get_default_query_reader()
    per_file_reports: list[tuple[str, dict]] = []

    async def _main() -> None:
        for f in all_tests_files:
            data = json.loads(f.read_text(encoding="utf-8"))
            name = data.get("name", f.stem)

            out_dir = out_base / name if is_dir_input else out_base
            out_dir.mkdir(parents=True, exist_ok=True)

            sep = "=" * 60
            print(f"\n{sep}")
            print(f"Processing: {f}")
            print(f"Output:     {out_dir}")

            # Step 1: Inference
            print("\n--- Step 1: Inference ---")
            await _run_one_all_tests(
                input_file=f,
                out_dir=out_dir,
                workers=workers,
                retry=retry,
                query_reader=query_reader,
                scene_per_train=scene_per_train,
                memory=memory,
                memory_proxy_port=memory_proxy_port,
                buffer_turns=buffer_turns,
            )

            # Step 2: Scoring
            print("\n--- Step 2: Scoring ---")
            run_scoring(input_path=str(f), result_dir=str(out_dir))

            # Step 3: Report
            print("\n--- Step 3: Report ---")
            # Auto-discover compaction_results.json from the same directory as
            # the all_tests.json file so a single `run` command covers the
            # full compact → infer → scoring → report pipeline.
            auto_compaction = f.parent / "compaction_results.json"
            has_compaction = auto_compaction.exists()
            compaction_path = str(auto_compaction) if has_compaction else None
            if compaction_path:
                print(f"[info] Auto-detected compaction results: {auto_compaction}")
            run_report(
                result_dir=str(out_dir),
                compaction_path=compaction_path,
                output_dir=out_dir,
            )

            # Collect report data for combined summary
            report_json_path = out_dir / "report.json"
            if report_json_path.exists():
                report_data = json.loads(report_json_path.read_text(encoding="utf-8"))
                per_file_reports.append((name, report_data))

    asyncio.run(_main())

    # Combined reports.md when multiple test sets were processed
    if len(per_file_reports) > 1:
        _generate_combined_reports(out_base, per_file_reports)

    print(f"\nRun complete. Results in: {out_base}")
