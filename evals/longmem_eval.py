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
    python longmem_eval.py --limit 1 --max-chunks 15 --max-questions 5 \
        --run-llm --mirix_config_path ./configs/0201c.yaml \
        --output_path results/0201c_longmem

Output lands in evals/results/longmem/<output_path>/ so it cannot collide
with the LoCoMo namespace used by main_eval.py.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from mirix_memory_system import MirixMemorySystem
from task_agent import TaskAgent

# LongMemEval rows in MemoryAgentBench are tagged metadata.source = "longmemeval_s*".
LONGMEM_SOURCE_PREFIX = "longmemeval_s"


def load_longmem_s(limit: Optional[int] = None) -> List[Dict]:
    """Load LongMemEval-S samples from HuggingFace MemoryAgentBench.

    Returns a list of dicts shaped as:
      {sample_id, context, questions, answers, question_types}
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
        source = meta.get("source")
        if not (isinstance(source, str) and source.startswith(LONGMEM_SOURCE_PREFIX)):
            continue
        idx = len(samples)
        samples.append(
            {
                "sample_id": f"longmem_s_{idx}",
                "context": row.get("context") or "",
                "questions": list(row.get("questions") or []),
                "answers": list(row.get("answers") or []),
                "question_types": list(meta.get("question_types") or []),
            }
        )
        if limit is not None and len(samples) >= limit:
            break

    if not samples:
        raise SystemExit(
            "No LongMemEval-S rows found in ai-hyz/MemoryAgentBench "
            f"(looked for metadata.source starting with {LONGMEM_SOURCE_PREFIX!r})."
        )
    return samples


def parse_sessions(context: str, max_chunk_chars: int = 4096) -> List[Dict]:
    """Parse a LongMemEval context into timestamped ingest chunks.

    The HF `context` field is the repr of a Python list shaped:
        ['Chat Time: 2022/11/17 (Thu) 12:04', [{'role','content'}, ...],
         'Chat Time: 2022/12/28 (Wed) 16:10', [ ... ], ...]
    i.e. alternating (chat-time string, message list) pairs — one pair per
    conversation session.

    The session structure is used ONLY to recover each session's real
    timestamp (the chat time). A whole session averages ~14k chars / ~3.6k
    tokens — far larger than the old 4096-char chunk — and feeding such large
    blocks to the LightRAG extractor measurably dilutes recall of small
    concrete facts (a counting list, a specific cake, a date). So each session
    is further split into <= `max_chunk_chars` chunks on message boundaries,
    and every sub-chunk inherits its session's `occurred_at`.

    Net effect: episodic agent still gets the correct year (occurred_at),
    AND the extractor sees small chunks again.

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
        """Split one session into <= max_chunk_chars chunks on message
        boundaries. Each chunk keeps the session header + occurred_at."""
        lines = _msg_lines(msgs)
        chunks: List[Dict] = []
        buf: List[str] = []
        buf_len = len(header)
        for ln in lines:
            # +1 for the newline join
            if buf and buf_len + len(ln) + 1 > max_chunk_chars:
                chunks.append({"occurred_at": occurred,
                               "text": header + "\n" + "\n".join(buf)})
                buf = []
                buf_len = len(header)
            # A single message longer than the budget still goes in alone.
            buf.append(ln)
            buf_len += len(ln) + 1
        if buf:
            chunks.append({"occurred_at": occurred,
                           "text": header + "\n" + "\n".join(buf)})
        return chunks or [{"occurred_at": occurred, "text": header}]

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


def measure_memory_size(sample_id: str) -> Dict:
    """Common-unit memory size: total stored characters + tokens.

    Both backends are measured on the SAME yardstick so no-graph (PG flat
    memory) and graph (Neo4j entities/relations) are directly comparable:
    we concatenate every stored text field and count chars/tokens.

      - PG flat memory: episodic summary+details, semantic name+summary+details
      - Neo4j graph:    entity descriptions + relation descriptions

    Also keeps the raw counts (rows / nodes / edges) for reference.
    """
    stats: Dict = {"unit": "chars+tokens"}
    pg_bin = "/usr/local/opt/postgresql@17/bin/psql"
    import subprocess

    def _pg(sql: str) -> str:
        try:
            out = subprocess.run(
                [pg_bin, "-h", "localhost", "-U", "mirix", "-d", "mirix", "-tAc", sql],
                capture_output=True, text=True, timeout=30,
                env={**os.environ, "PGPASSWORD": "mirix"},
            )
            return out.stdout.strip()
        except Exception:
            return ""

    # --- PG flat memory: rows + concatenated text size ---
    flat: Dict = {}
    flat_chars = 0
    for table, cols in (
        ("episodic_memory", "coalesce(summary,'')||coalesce(details,'')"),
        ("semantic_memory", "coalesce(name,'')||coalesce(summary,'')||coalesce(details,'')"),
    ):
        row_n = _pg(f"SELECT count(*) FROM {table} WHERE user_id='{sample_id}';")
        chars = _pg(f"SELECT coalesce(sum(length({cols})),0) FROM {table} WHERE user_id='{sample_id}';")
        n = int(row_n) if row_n.isdigit() else 0
        c = int(chars) if chars.lstrip('-').isdigit() else 0
        flat[table] = {"rows": n, "chars": c}
        flat_chars += c
    stats["flat"] = flat
    stats["flat_total_chars"] = flat_chars

    # --- Neo4j graph: node/edge counts + concatenated description size ---
    try:
        from neo4j import GraphDatabase
        from mirix.settings import settings
        driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
        with driver.session(database=settings.neo4j_database) as ses:
            rec = ses.run(
                """
                MATCH (n) WHERE n.user_id=$uid
                WITH count(n) AS nodes,
                     sum(size(coalesce(n.description,'')) + size(coalesce(n.summary,''))) AS node_chars
                RETURN nodes, node_chars
                """, uid=sample_id
            ).single()
            erec = ses.run(
                """
                MATCH (a)-[r]->(b) WHERE a.user_id=$uid
                RETURN count(r) AS edges,
                       sum(size(coalesce(r.description,''))) AS edge_chars
                """, uid=sample_id
            ).single()
        driver.close()
        node_chars = (rec["node_chars"] or 0) if rec else 0
        edge_chars = (erec["edge_chars"] or 0) if erec else 0
        stats["graph"] = {
            "nodes": (rec["nodes"] or 0) if rec else 0,
            "edges": (erec["edges"] or 0) if erec else 0,
            "node_chars": node_chars,
            "edge_chars": edge_chars,
        }
        stats["graph_total_chars"] = node_chars + edge_chars
    except Exception as exc:
        stats["graph"] = {"error": str(exc)}
        stats["graph_total_chars"] = 0

    # Common-unit summary: whichever backend stored memory, in chars + tokens.
    total_chars = max(stats["flat_total_chars"], stats["graph_total_chars"])
    stats["total_chars"] = total_chars
    stats["total_tokens"] = total_chars // 4  # cheap estimate; exact below if small
    return stats


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
    parser.add_argument("--mirix_config_path", type=Path, default=None,
                        help="Path to Mirix config file.")
    args = parser.parse_args()

    items = load_longmem_s(limit=args.limit)
    print(f"[longmem_eval] loaded {len(items)} LongMemEval-S conversation(s)")

    mirix_client_id = os.environ.get("MIRIX_CLIENT_ID", "mirix-eval-client")
    mirix_org_id = os.environ.get("MIRIX_ORG_ID", "mirix-eval-org")

    # Keep LongMemEval output in its own namespace, away from LoCoMo's.
    longmem_root = Path(__file__).resolve().parent / "results" / "longmem"
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
            all_memories = memory_system.list_all_memories(limit=0)
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
