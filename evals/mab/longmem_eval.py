"""Evaluate Mirix memory on MemoryAgentBench LongMemEval-S.

Sits alongside main_eval.py (LoCoMo runner). Reuses the same scaffolding —
MirixMemorySystem, TaskAgent, server-side token tracker — and writes the same
per-sample JSON schema, so organize_results.py works on its output unchanged.

Only the data layer differs: LongMemEval-S comes from the HuggingFace dataset
`ai-hyz/MemoryAgentBench` (split `Accurate_Retrieval`, rows whose
metadata.source starts with `longmemeval_s`). Each row is one long context
(~1.6M chars) plus a list of questions / answers / per-question types.

The long context is a list of conversation sessions, each prefixed with a
"Chat Time". It is parsed into one chunk per session and ingested chunk by
chunk, with each session's chat time passed to MIRIX as `occurred_at` — so the
episodic agent anchors the real year instead of guessing it.

Usage (smoke test — 1 conv, 15 chunks, 5 questions):
    python mab/longmem_eval.py --limit 1 --max-chunks 15 --max-questions 5 \
        --run-llm --mirix_config_path ./configs/0201c.yaml \
        --output_path results/0201c_longmem

Output lands in evals/results/longmem/<output_path>/ so it cannot collide
with the LoCoMo namespace used by main_eval.py.
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

# LongMemEval rows in MemoryAgentBench are tagged metadata.source = "longmemeval_s*".
# Default mirrors the official MAB judge CLI (--dataset longmemeval_s*).
# Exact-match (==) rather than startswith — matches official
# llm_based_eval/longmem_qa_evaluate.py behaviour.
DEFAULT_LONGMEM_SOURCE = "longmemeval_s*"


def load_longmem_s(
    source: str = DEFAULT_LONGMEM_SOURCE,
    limit: Optional[int] = None,
) -> List[Dict]:
    """Load LongMemEval-S samples from HuggingFace MemoryAgentBench.

    Returns a list of dicts shaped as:
      {sample_id, context, questions, answers, question_types, question_ids}

    ``question_ids`` carry the HF metadata IDs (e.g. ``"a4996e51"``,
    ``"edced276_abs"``). The ``_abs`` suffix is what the official judge
    uses to route abstention questions to a different prompt.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "The `datasets` package is required for LongMemEval-S. "
            "Install it with: uv pip install datasets"
        ) from exc

    ds = load_dataset("ai-hyz/MemoryAgentBench", split="Accurate_Retrieval")

    samples: List[Dict] = []
    for row in ds:
        meta = row.get("metadata") or {}
        if meta.get("source") != source:
            continue
        idx = len(samples)
        samples.append(
            {
                "sample_id": f"longmem_s_{idx}",
                "context": row.get("context") or "",
                "questions": list(row.get("questions") or []),
                "answers": list(row.get("answers") or []),
                "question_types": list(meta.get("question_types") or []),
                "question_ids": list(meta.get("question_ids") or []),
            }
        )
        if limit is not None and len(samples) >= limit:
            break

    if not samples:
        raise SystemExit(
            f"No rows found in ai-hyz/MemoryAgentBench (Accurate_Retrieval) "
            f"with metadata.source == {source!r}."
        )
    return samples


from _chunking import DEFAULT_CHUNK_TOKENS, chunk_text_into_sentences


def parse_sessions(context: str, max_chunk_tokens: int = DEFAULT_CHUNK_TOKENS) -> List[Dict]:
    """Parse a LongMemEval context into timestamped ingest chunks.

    The HF ``context`` field is the repr of a Python list shaped:
        ['Chat Time: 2022/11/17 (Thu) 12:04', [{'role','content'}, ...],
         'Chat Time: 2022/12/28 (Wed) 16:10', [ ... ], ...]
    i.e. alternating (chat-time string, message list) pairs — one pair
    per conversation session.

    Each session is then chunked with the shared sentence-aware,
    token-budgeted chunker (4096 gpt-4o-mini tokens, aligned with the
    official MAB chunker). Most LongMemEval-S sessions (~14k chars /
    ~3.6k tokens) fit in a single chunk now, but the
    ``chunk_text_into_sentences`` call still defends against the rare
    long session. Every sub-chunk inherits its session's ``occurred_at``
    so the episodic agent's temporal reasoning stays anchored.

    Returns: list of {occurred_at: Optional[str], text: str}.
    """
    import ast
    import re
    from datetime import datetime

    try:
        parsed = ast.literal_eval(context)
    except (ValueError, SyntaxError):
        # Fallback: treat the whole context as one undated chunk.
        return [{"occurred_at": None, "text": context}]

    if not isinstance(parsed, list):
        return [{"occurred_at": None, "text": str(context)}]

    def _parse_chat_time(s: str) -> Optional[str]:
        # "Chat Time: 2022/11/17 (Thu) 12:04" -> "2022-11-17T12:04:00"
        m = re.search(r"(\d{4})/(\d{1,2})/(\d{1,2}).*?(\d{1,2}):(\d{2})", s)
        if not m:
            return None
        y, mo, d, hh, mm = (int(x) for x in m.groups())
        try:
            return datetime(y, mo, d, hh, mm).isoformat()
        except ValueError:
            return None

    def _msg_lines(msgs) -> List[str]:
        lines = []
        for msg in msgs:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).strip()
            content = str(msg.get("content", "")).strip()
            lines.append(f"{role}: {content}")
        return lines

    def _split_session(header: str, msgs, occurred: Optional[str]) -> List[Dict]:
        """Sentence-tokenize the session and inherit the session's
        ``occurred_at`` on every resulting sub-chunk. Header is
        prepended to the first text blob so the chunker can decide
        whether it fits alongside body content."""
        body_lines = _msg_lines(msgs)
        text = header + "\n" + "\n".join(body_lines) if body_lines else header
        pieces = chunk_text_into_sentences(text, chunk_size=max_chunk_tokens)
        if not pieces:
            return [{"occurred_at": occurred, "text": header}]
        return [{"occurred_at": occurred, "text": p} for p in pieces]

    sessions: List[Dict] = []
    i = 0
    while i < len(parsed):
        item = parsed[i]
        if isinstance(item, str) and "Chat Time" in item and i + 1 < len(parsed) \
                and isinstance(parsed[i + 1], list):
            occurred = _parse_chat_time(item)
            sessions.extend(_split_session(item, parsed[i + 1], occurred))
            i += 2
        else:
            # Unexpected shape — keep it as an undated chunk rather than drop.
            if isinstance(item, list):
                for ln in _msg_lines(item):
                    sessions.append({"occurred_at": None, "text": ln})
            elif isinstance(item, str):
                sessions.append({"occurred_at": None, "text": item})
            i += 1
    return sessions


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
    parser = argparse.ArgumentParser(description="Evaluate Mirix memory on LongMemEval-S.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of conversations (samples) to evaluate.")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="Limit session chunks ingested per sample (smoke-test knob).")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit number of questions per sample.")
    parser.add_argument("--run-llm", action="store_true", default=True,
                        help="Call the LLM to answer questions.")
    parser.add_argument("--output_path", type=Path, default=Path("longmem_run"),
                        help="Output sub-folder, resolved under evals/results/longmem/.")
    parser.add_argument(
        "--source",
        type=str,
        default=DEFAULT_LONGMEM_SOURCE,
        help=(
            "HF metadata.source to load (exact match). Default "
            f"{DEFAULT_LONGMEM_SOURCE!r} mirrors the official MAB judge "
            "CLI (--dataset longmemeval_s*)."
        ),
    )
    parser.add_argument("--mirix_config_path", type=Path, default=None,
                        help="Path to Mirix config file.")
    args = parser.parse_args()

    items = load_longmem_s(source=args.source, limit=args.limit)
    print(f"[longmem_eval] loaded {len(items)} LongMemEval-S conversation(s)")

    mirix_client_id = os.environ.get("MIRIX_CLIENT_ID", "mirix-eval-client")
    mirix_org_id = os.environ.get("MIRIX_ORG_ID", "mirix-eval-org")

    # Keep LongMemEval output in its own namespace, away from LoCoMo's.
    longmem_root = EVALS_DIR / "results" / "longmem"
    if args.output_path.is_absolute():
        print(f"[longmem_eval] WARNING: --output_path is absolute ({args.output_path}); "
              "writing outside evals/results/longmem/ namespace.")
        output_path = args.output_path
    else:
        output_path = longmem_root / args.output_path
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"[longmem_eval] writing per-sample results to {output_path}")

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

        # ---- ingest: one chunk per conversation session, WITH timestamps ----
        chunks = parse_sessions(item["context"])
        if args.max_chunks is not None:
            chunks = chunks[: args.max_chunks]
        dated = sum(1 for c in chunks if c["occurred_at"])
        print(f"[longmem_eval] {sample_id}: ingesting {len(chunks)} session chunk(s) "
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
        print(f"[longmem_eval] {sample_id}: memory_stats = {sample_result['memory_stats']}")
        save_sample_result(sample_path, sample_result)

        # ---- QA ----
        questions = item["questions"]
        answers = item["answers"]
        qtypes = item["question_types"]
        qids = item.get("question_ids") or []
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
