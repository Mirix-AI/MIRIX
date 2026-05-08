"""Driver: run metaclaw-bench day01..day03 against MIRIX as evolver+retriever.

Run from the MIRIX repo root:

    python -m evals.metaclaw.run_3day_eval                       # full 3 days
    python -m evals.metaclaw.run_3day_eval --days day01          # just day01
    python -m evals.metaclaw.run_3day_eval --max-rounds 1        # smoke
    python -m evals.metaclaw.run_3day_eval --dry-run             # no LLM, no MIRIX
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import os
import shutil
import sys
import uuid
from pathlib import Path

logger = logging.getLogger("evals.metaclaw")

REPO_ROOT = Path(__file__).resolve().parents[2]
METACLAW_BENCH = REPO_ROOT / "third_party" / "MetaClaw" / "benchmark" / "data" / "metaclaw-bench"
EVAL_DIR = METACLAW_BENCH / "eval"
WORKSPACE_SRC = METACLAW_BENCH / "workspaces" / "shared"
SCORE_SCRIPT_DIR = REPO_ROOT / "third_party" / "MetaClaw" / "scripts"
DEFAULT_DAYS = ["day01", "day02", "day03"]


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )


def _run_id() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def _prepare_workspace(run_dir: Path) -> Path:
    """Copy workspaces/shared/ into run_dir/workspace/. Carries across all days."""
    ws = run_dir / "workspace"
    if ws.exists():
        shutil.rmtree(ws)
    shutil.copytree(WORKSPACE_SRC, ws)
    # Also copy bench scripts so eval.command lines like
    # `python scripts/check_iso8601.py` resolve from the workspace cwd.
    scripts_dst = ws / "scripts"
    if not scripts_dst.exists() and SCORE_SCRIPT_DIR.exists():
        shutil.copytree(SCORE_SCRIPT_DIR, scripts_dst)
    return ws


def _load_questions(day: str) -> dict:
    return json.loads((EVAL_DIR / day / "questions.json").read_text())


def _expected_feedback(round_obj: dict, outcome: str) -> str:
    fb = round_obj.get("feedback", {})
    if isinstance(fb, dict):
        return fb.get("correct", "") if outcome == "pass" else fb.get("incorrect", "")
    return str(fb)


async def _amain(args: argparse.Namespace) -> int:
    _setup_logging(args.log_level)

    from evals.metaclaw.format_adapter import RoundResult
    from evals.metaclaw.llm_config_helpers import (
        DEFAULT_CHAT_MODEL,
        OPENROUTER_BASE_URL,
        assert_openrouter_env,
    )
    from evals.metaclaw.mirix_client import MirixClient
    from evals.metaclaw.mirix_skill_evolver import MirixSkillEvolver
    from evals.metaclaw.mirix_skill_manager import MirixSkillManager
    from evals.metaclaw.round_runner import RunnerConfig, run_round

    if not args.dry_run:
        assert_openrouter_env()

    days = args.days or DEFAULT_DAYS
    run_id = _run_id()
    run_dir = REPO_ROOT / "evals" / "metaclaw" / "reports" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run %s — output dir %s", run_id, run_dir)

    workspace = _prepare_workspace(run_dir)
    logger.info("Workspace prepared at %s", workspace)

    # MIRIX wiring
    mirix = None
    evolver = None
    skill_mgr = None
    if not args.dry_run:
        mirix = MirixClient(
            base_url=args.mirix_url,
            user_id=args.user_id,
            timeout=args.mirix_timeout,
        )
        if not await mirix.health():
            logger.error("MIRIX server not reachable at %s. "
                         "Start it with: python scripts/start_server.py --port 8531",
                         args.mirix_url)
            return 2
        evolver = MirixSkillEvolver(mirix_client=mirix)
        skill_mgr = MirixSkillManager(mirix_client=mirix)

    # OpenAI client for the agent loop
    openai_client = None
    chat_model = os.environ.get("EVAL_CHAT_MODEL", DEFAULT_CHAT_MODEL)
    if not args.dry_run:
        from openai import OpenAI
        openai_client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_API_BASE", OPENROUTER_BASE_URL),
        )

    summary = {"run_id": run_id, "days": [], "started_at": dt.datetime.now().isoformat()}

    for day in days:
        logger.info("=== %s ===", day)
        q = _load_questions(day)
        rounds = q.get("rounds", [])
        if args.max_rounds:
            rounds = rounds[: args.max_rounds]
        round_results: list[RoundResult] = []

        for r in rounds:
            round_id = r["id"]
            round_type = r["type"]
            question = r["question"]
            eval_block = r.get("eval", {})

            if args.dry_run:
                # No LLM, no MIRIX — produce a deterministic stub for plumbing tests
                from evals.metaclaw.format_adapter import RoundResult as RR
                rr = RR(
                    round_id=round_id, round_type=round_type, question=question,
                    final_answer="(dry-run)", reward=0.0, eval_outcome="fail",
                    feedback=_expected_feedback(r, "fail"), error="dry_run",
                )
                round_results.append(rr)
                logger.info("[%s/%s] dry-run", day, round_id)
                continue

            skills = skill_mgr.retrieve(question, top_k=args.top_k)
            logger.info("[%s/%s] retrieved %d skills", day, round_id, len(skills))
            cfg = RunnerConfig(
                chat_model=chat_model, workspace=workspace,
                max_turns=args.max_turns, wallclock_cap_s=args.wallclock_cap,
            )
            rr = run_round(
                openai_client=openai_client, cfg=cfg,
                round_id=round_id, round_type=round_type,
                question=question, eval_block=eval_block, skills=skills,
            )
            rr.feedback = _expected_feedback(r, rr.eval_outcome)
            round_results.append(rr)
            logger.info("[%s/%s] outcome=%s reward=%s",
                        day, round_id, rr.eval_outcome, rr.reward)

        # Day-end evolve
        evolve_status = "skipped"
        diff_summary = {"created": [], "edited": [], "deleted": []}
        if not args.dry_run and not args.no_evolve:
            try:
                metaclaw_skills = await evolver.evolve(round_results, current_skills={})
                evolve_status = "ok"
                diff_summary = {
                    "produced_skills": [s["name"] for s in metaclaw_skills],
                }
                logger.info("[%s] evolve produced %d skills: %s",
                            day, len(metaclaw_skills),
                            [s["name"] for s in metaclaw_skills])
            except Exception as e:
                evolve_status = f"failed:{type(e).__name__}:{e}"
                logger.warning("[%s] evolve failed: %s", day, e)

        # Per-day metrics
        n = len(round_results)
        passed = sum(1 for r in round_results if r.reward >= 1.0)
        per_round = [
            {"id": r.round_id, "type": r.round_type, "outcome": r.eval_outcome,
             "reward": r.reward, "error": r.error}
            for r in round_results
        ]
        day_metrics = {
            "day": day, "n_rounds": n, "n_passed": passed,
            "pass_rate": (passed / n) if n else 0.0,
            "per_round": per_round,
            "evolve_status": evolve_status, "evolve_diff": diff_summary,
        }
        (run_dir / f"{day}_metrics.json").write_text(
            json.dumps(day_metrics, indent=2, default=str)
        )
        summary["days"].append(day_metrics)

    summary["finished_at"] = dt.datetime.now().isoformat()
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    _write_summary_md(run_dir, summary)
    logger.info("Done. Summary at %s/summary.md", run_dir)

    if mirix is not None:
        await mirix.aclose()
    return 0


def _write_summary_md(run_dir: Path, summary: dict) -> None:
    lines = ["# MIRIX × MetaClaw Eval Summary", "",
             f"Run id: `{summary['run_id']}`",
             f"Started: {summary['started_at']}",
             f"Finished: {summary['finished_at']}", "",
             "## Per-day pass rate", "",
             "| Day | Rounds | Passed | Pass rate | Evolve | Skills produced |",
             "|---|---|---|---|---|---|"]
    for d in summary["days"]:
        produced = d["evolve_diff"].get("produced_skills", [])
        lines.append(
            f"| {d['day']} | {d['n_rounds']} | {d['n_passed']} | "
            f"{d['pass_rate']:.2f} | {d['evolve_status']} | "
            f"{', '.join(produced) or '—'} |"
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", nargs="+", choices=["day01", "day02", "day03"],
                   help="Subset of days to run (default: all three).")
    p.add_argument("--max-rounds", type=int, default=0,
                   help="Cap rounds per day (0 = no cap; useful for smoke).")
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument("--wallclock-cap", type=float, default=300.0)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--user-id", type=str, default="eval-metaclaw-3day")
    p.add_argument("--mirix-url", type=str, default="http://127.0.0.1:8531")
    p.add_argument("--mirix-timeout", type=float, default=600.0)
    p.add_argument("--no-evolve", action="store_true",
                   help="Skip day-end evolve calls (debug).")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip all LLM and MIRIX calls; emit stub metrics for plumbing tests.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:]) if argv is None else argv)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
