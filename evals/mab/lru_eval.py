"""Evaluate Mirix memory on MemoryAgentBench Long_Range_Understanding (LRU).

Sister script to ruler_eval.py — same scaffolding (MirixMemorySystem,
TaskAgent, server-side token tracker, per-sample JSON schema, char-
based chunking, _eval_db helpers) so organize_results.py works on the
output unchanged.

LRU rows come from `ai-hyz/MemoryAgentBench`, split
``Long_Range_Understanding``, with two sources:

  - ``infbench_sum_eng_shots2`` (100 rows, novels) — one long story
    per row (~1.7M chars), one question per row ("Write a 1000-1200
    word summary..."), answer is a multi-paragraph reference summary.
    Score with ``--judge mab_summary`` (gpt-4o-2024-05-13, fluency +
    recall over keypoints + precision over sentences → F1).
  - ``detective_qa`` (10 rows) — multiple-choice mystery QA. Score
    with ``--judge substring`` (the gold answers are short option
    strings like "C. The Brandt couple").

For ``infbench_sum_eng_shots2`` we propagate ``keypoints`` and the
``qa_pair_id`` from ``metadata`` into each record so
``llm_judge_mab_summary`` has what it needs without a re-fetch.

Like RULER, the context carries no timestamps so ``occurred_at`` is
always None. Char-based 4096 chunking matches ruler_eval; the official
MAB chunker is sentence-aware token-based ~20k chars but that exposed
the httpx RetryTransport timeout; we trade leaderboard parity for
runs that finish in this house.

Usage (smoke test — 1 row, 3 chunks):
    python mab/lru_eval.py --limit 1 --max-chunks 3 \\
        --run-llm --mirix_config_path ./configs/mab.yaml \\
        --output_path smoke_lru

Output lands in evals/results/lru/<output_path>/.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

EVALS_DIR = Path(__file__).resolve().parent.parent
if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

from mirix_memory_system import MirixMemorySystem
from task_agent import TaskAgent

# LRU sources in MemoryAgentBench (split Long_Range_Understanding):
#   infbench_sum_eng_shots2 — novel summarization (100 rows)   ← default
#   detective_qa            — multiple-choice mystery QA (10 rows)
# Default points at infbench; swap via --source.
DEFAULT_LRU_SOURCE = "infbench_sum_eng_shots2"


def load_lru(
    source: str = DEFAULT_LRU_SOURCE,
    limit: Optional[int] = None,
) -> List[Dict]:
    """Load Long_Range_Understanding samples from HuggingFace.

    Returns a list of dicts shaped as:
      {sample_id, context, questions, answers, keypoints, qa_pair_ids}

    keypoints/qa_pair_ids are populated from row metadata for
    ``infbench_sum_eng_shots2`` and propagated into each record so the
    mab_summary judge can score without re-fetching HF. For
    ``detective_qa`` they're empty lists (substring judge ignores them).
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "The `datasets` package is required for LRU. "
            "Install it with: uv pip install datasets"
        ) from exc

    ds = load_dataset("ai-hyz/MemoryAgentBench", split="Long_Range_Understanding")

    # Namespace sample_id by source so infbench / detective rows can
    # share the same Postgres DB without their user_id rows getting
    # mixed up (both subsets number from 0).
    source_tag = {
        "infbench_sum_eng_shots2": "infbench",
        "detective_qa": "detective",
    }.get(source, source.lower())

    samples: List[Dict] = []
    for row in ds:
        meta = row.get("metadata") or {}
        if meta.get("source") != source:
            continue
        idx = len(samples)
        samples.append(
            {
                "sample_id": f"{source_tag}_{idx}",
                "context": row.get("context") or "",
                "questions": list(row.get("questions") or []),
                "answers": list(row.get("answers") or []),
                # MAB summarization judge inputs (infbench only).
                # detective_qa rows leave these as empty / from-meta.
                "keypoints": list(meta.get("keypoints") or []),
                "qa_pair_ids": list(meta.get("qa_pair_ids") or []),
                # Empty lists — kept for shape compatibility with the
                # downstream record builder; abstention prompt is never
                # triggered on LRU.
                "question_types": [],
                "question_ids": [],
            }
        )
        if limit is not None and len(samples) >= limit:
            break

    if not samples:
        raise SystemExit(
            f"No rows found in ai-hyz/MemoryAgentBench (Long_Range_Understanding) "
            f"with metadata.source == {source!r}. "
            f"Try {DEFAULT_LRU_SOURCE!r} or 'detective_qa'."
        )
    return samples


from _chunking import DEFAULT_CHUNK_TOKENS, chunk_text_into_sentences


def parse_context(context: str, max_chunk_tokens: int = DEFAULT_CHUNK_TOKENS) -> List[Dict]:
    """Sentence-aware, token-budgeted chunking for an LRU context.

    Delegates to the shared ``chunk_text_into_sentences`` (NLTK ``punkt``
    + tiktoken ``gpt-4o-mini``, 4096-token budget) — matches the
    official MAB chunker so infbench / detective scores are directly
    leaderboard-comparable.

    LRU contexts have no timestamps, so ``occurred_at`` is always None.

    Returns: list of {occurred_at: None, text: str}.
    """
    return [{"occurred_at": None, "text": c}
            for c in chunk_text_into_sentences(context, chunk_size=max_chunk_tokens)]


def flatten_answer(answer) -> Optional[str]:
    """LongMemEval answers are list-of-list (e.g. ['50']). Flatten to a string."""
    if answer is None:
        return None
    if isinstance(answer, list):
        flat = []
        for a in answer:
            if isinstance(a, list):
                flat.extend(str(x) for x in a)
            else:
                flat.append(str(a))
        return "; ".join(flat) if flat else None
    return str(answer)


from _eval_db import measure_memory_size, dump_memories


def load_sample_result(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        return None
    return None


def save_sample_result(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def print_qa(qidx: int, question: str, expected: Optional[str], predicted: Optional[str]) -> None:
    print(f"[{qidx}] question: {str(question)[:120]}")
    print(f"[{qidx}] expected_answer: {expected}")
    print(f"[{qidx}] predicted_answer: {str(predicted)[:200] if predicted else predicted}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Mirix memory on MAB Long_Range_Understanding (infbench / detective_qa).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples to evaluate.")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="Limit ingest chunks per sample (smoke-test knob).")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit number of questions per sample.")
    parser.add_argument("--run-llm", action="store_true", default=True,
                        help="Call the LLM to answer questions.")
    parser.add_argument("--output_path", type=Path, default=Path("lru_run"),
                        help="Output sub-folder, resolved under evals/results/lru/.")
    parser.add_argument(
        "--source",
        type=str,
        default=DEFAULT_LRU_SOURCE,
        help=(
            "HF metadata.source to load (exact match). Default "
            f"{DEFAULT_LRU_SOURCE!r}. Swap to 'detective_qa' for the "
            "multiple-choice mystery subset."
        ),
    )
    parser.add_argument("--mirix_config_path", type=Path, default=None,
                        help="Path to Mirix config file.")
    args = parser.parse_args()

    items = load_lru(source=args.source, limit=args.limit)
    print(f"[lru_eval] loaded {len(items)} LRU conversation(s) from source {args.source!r}")

    mirix_client_id = os.environ.get("MIRIX_CLIENT_ID", "mirix-eval-client")
    mirix_org_id = os.environ.get("MIRIX_ORG_ID", "mirix-eval-org")

    # Keep LRU output in its own namespace, away from RULER / LongMemEval / LoCoMo.
    lru_root = EVALS_DIR / "results" / "lru"
    if args.output_path.is_absolute():
        print(f"[lru_eval] WARNING: --output_path is absolute ({args.output_path}); "
              "writing outside evals/results/lru/ namespace.")
        output_path = args.output_path
    else:
        output_path = lru_root / args.output_path
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"[lru_eval] writing per-sample results to {output_path}")

    import httpx
    server_base = "http://127.0.0.1:8531"

    def _reset_tokens():
        try:
            httpx.post(f"{server_base}/debug/token_stats/reset", timeout=10)
        except Exception:
            pass

    def _snapshot_tokens():
        try:
            r = httpx.get(f"{server_base}/debug/token_stats", timeout=10)
            return r.json().get("stats", {})
        except Exception:
            return {}

    def _sum_tokens(stats):
        s = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}
        for v in stats.values():
            for k in s:
                s[k] += v.get(k, 0)
        return s

    for item in items:
        sample_id = item["sample_id"]
        sample_path = output_path / f"{sample_id}.json"

        task_agent = (
            TaskAgent(mirix_config_path=str(args.mirix_config_path),
                      client_id=mirix_client_id, org_id=mirix_org_id, user_id=sample_id)
            if args.run_llm else None
        )

        sample_result = load_sample_result(sample_path) or {
            "sample_id": sample_id,
            "timings": {"add_chunk": {}, "wrap_user_prompt": {}, "answer": {}},
            "responses": {},
            "records": {},
        }
        sample_result.setdefault("sample_id", sample_id)

        memory_system = MirixMemorySystem(
            user_id=sample_id,
            mirix_config_path=str(args.mirix_config_path),
            client=task_agent.mirix_client if task_agent else None,
        )

        _reset_tokens()

        # ---- ingest: paragraph-bundled chunks of <= 4096 chars ----
        chunks = parse_context(item["context"])
        if args.max_chunks is not None:
            chunks = chunks[: args.max_chunks]
        dated = sum(1 for c in chunks if c["occurred_at"])
        print(f"[lru_eval] {sample_id}: ingesting {len(chunks)} context chunk(s) "
              f"({dated} with timestamps)")

        for idx, chunk in enumerate(chunks, start=1):
            idx_key = str(idx)
            if idx_key in sample_result["responses"]:
                continue
            start = time.perf_counter()
            response = memory_system.add_chunk(
                chunk["text"], raw_input=chunk["text"],
                occurred_at=chunk["occurred_at"],
            )
            elapsed = time.perf_counter() - start
            sample_result["responses"][idx_key] = {
                "type": "add_chunk",
                "chunk_index": idx,
                "question_index": None,
                "occurred_at": chunk["occurred_at"],
                "response": response,
            }
            sample_result["timings"]["add_chunk"][idx_key] = elapsed
            save_sample_result(sample_path, sample_result)

        build_stats = _snapshot_tokens()
        sample_result["token_stats"] = {"build_raw": build_stats, "build_sum": _sum_tokens(build_stats)}
        save_sample_result(sample_path, sample_result)

        # Memory size in a common unit (stored chars) so no-graph (flat PG) and
        # graph (Neo4j) runs are directly comparable. See measure_memory_size().
        sample_result["memory_stats"] = measure_memory_size(sample_id)
        print(f"[lru_eval] {sample_id}: memory_stats = {sample_result['memory_stats']}")
        save_sample_result(sample_path, sample_result)

        # ---- QA ----
        questions = item["questions"]
        answers = item["answers"]
        qtypes = item["question_types"]
        qids = item.get("question_ids") or []
        # MAB summarization judge inputs (infbench only). qa_pair_ids is
        # parallel to questions for detective_qa; for infbench_sum the
        # whole row is one qa_pair so we just reuse qa_pair_ids[0].
        keypoints = item.get("keypoints") or []
        qa_pair_ids = item.get("qa_pair_ids") or []
        if args.max_questions is not None:
            questions = questions[: args.max_questions]

        for qidx, question in enumerate(questions, start=1):
            qidx_key = str(qidx)
            if qidx_key in sample_result["records"]:
                rec = sample_result["records"][qidx_key]
                print_qa(qidx, rec.get("question", ""),
                         rec.get("expected_answer"), rec.get("predicted_answer"))
                continue

            expected_answer = flatten_answer(answers[qidx - 1] if qidx - 1 < len(answers) else None)
            # category: organize_results.py groups metrics by this. LongMemEval
            # uses string question types (multi-session, temporal-reasoning, ...).
            category = qtypes[qidx - 1] if qidx - 1 < len(qtypes) else None

            start = time.perf_counter()
            input_messages = memory_system.wrap_user_prompt(question)
            sample_result["timings"]["wrap_user_prompt"][qidx_key] = time.perf_counter() - start

            predicted = None
            message_trace = None
            usage_trace = None
            usage_total = None
            if task_agent:
                start = time.perf_counter()
                trace = task_agent.answer(input_messages, user_id=sample_id)
                predicted = trace.get("answer")
                message_trace = trace.get("messages")
                usage_trace = trace.get("usage")
                usage_total = trace.get("usage_total")
                sample_result["timings"]["answer"][qidx_key] = time.perf_counter() - start

            sample_result["records"][qidx_key] = {
                "sample_id": sample_id,
                "question_index": qidx,
                # HF metadata.question_ids[qidx-1]. Carries the optional
                # ``_abs`` suffix that routes abstention questions to the
                # MAB judge's abstention prompt.
                "question_id": qids[qidx - 1] if qidx - 1 < len(qids) else None,
                # Pass-through for organize_results.judge_task_mab_summary.
                "qa_pair_id": (
                    qa_pair_ids[qidx - 1] if qidx - 1 < len(qa_pair_ids)
                    else (qa_pair_ids[0] if qa_pair_ids else None)
                ),
                "keypoints": keypoints,
                "question": question,
                "expected_answer": expected_answer,
                "evidence": None,
                "category": category,
                "input_messages": input_messages,
                "predicted_answer": predicted,
                "messages": message_trace,
                "usage": usage_trace,
                "usage_total": usage_total,
            }
            save_sample_result(sample_path, sample_result)
            print_qa(qidx, question, expected_answer, predicted)

        try:
            all_memories = dump_memories(sample_id)
        except Exception as exc:
            all_memories = {"success": False, "error": str(exc), "user_id": sample_id}
        with (output_path / f"{sample_id}_memories.json").open("w", encoding="utf-8") as handle:
            json.dump(all_memories, handle, ensure_ascii=False, indent=2)

        post_qa_stats = _snapshot_tokens()
        post_qa_sum = _sum_tokens(post_qa_stats)
        build_sum = sample_result.get("token_stats", {}).get("build_sum", {})
        query_sum = {
            k: max(post_qa_sum.get(k, 0) - build_sum.get(k, 0), 0)
            for k in ("prompt", "completion", "total", "calls")
        }
        sample_result.setdefault("token_stats", {})
        sample_result["token_stats"]["query_raw"] = post_qa_stats
        sample_result["token_stats"]["query_sum"] = query_sum
        save_sample_result(sample_path, sample_result)


if __name__ == "__main__":
    main()
