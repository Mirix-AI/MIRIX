import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from mirix_memory_system import MirixMemorySystem
from task_agent import TaskAgent


instructions = """Instructions:

1. Carefully analyze all utterances from both speakers.
2. The conversation has a timestamp, but the events mentioned in the conversation may have different timestamps. You have to extract the exact date of the mentioned events. Remember that "mentioned at" is not the same as "occurred at" so this has to be noted in the memories.
3. If there is a question about time references (like "last year", "two months ago", etc.), calculate the actual date based on the memory timestamp. For example, if a memory from 4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
4. Always convert relative time references to specific dates, months, or years. For example, convert "last year" to "2022" or "two months ago" to "March 2023" based on the conversation timestamp.
5. Focus only on the content of the memories from both speakers. Do not confuse character names mentioned in memories with the actual users who created those memories.
6. You are supposed to extract the event/fact/semantic knowledge from the conversation. For example, if the conversation happens at 2023 and the conversation says that "John went to India last year", then you should save the fact that "John went to India in 2022". Similarly for all other kinds of memories.
7. Make sure to extract the facts about the characters, such as their name, age, gender, occupation, hometown, etc."""

def load_locomo(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    for key in ("data", "items", "examples", "records"):
        if key in data and isinstance(data[key], list):
            return data[key]
    raise ValueError(f"Unsupported dataset format in {path}")


def iter_sessions(conversation: Dict) -> Iterable[Dict]:
    session_numbers = []
    for key, value in conversation.items():
        match = re.match(r"^session_(\d+)$", key)
        if match and isinstance(value, list):
            session_numbers.append(int(match.group(1)))
    for number in sorted(session_numbers):
        yield {
            "number": number,
            "date_time": conversation.get(f"session_{number}_date_time"),
            "turns": conversation.get(f"session_{number}", []),
        }


def format_session_chunk(session: Dict, date_time: str) -> str:

    header = f"Session {session['number']}"
    if session.get("date_time"):
        header += f" ({session['date_time']})"
    lines = [f"You have access to the conversation between two speakers. The conversation is timestamped at {date_time}.\n"]
    lines.append(instructions)
    lines.append(header)
    for turn in session.get("turns", []):
        speaker = turn.get("speaker", "").strip()
        text = turn.get("text", "").strip()
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)

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


def normalize_sample_result(sample_result: Dict) -> Dict:
    timings = sample_result.setdefault(
        "timings",
        {"add_chunk": {}, "wrap_user_prompt": {}, "answer": {}},
    )
    for key in ("add_chunk", "wrap_user_prompt", "answer"):
        if isinstance(timings.get(key), list):
            timings[key] = {
                str(idx): value for idx, value in enumerate(timings[key], start=1)
            }
        elif not isinstance(timings.get(key), dict):
            timings[key] = {}

    responses = sample_result.setdefault("responses", {})
    if isinstance(responses, list):
        responses_dict: Dict[str, Dict] = {}
        for entry in responses:
            if isinstance(entry, dict):
                chunk_index = entry.get("chunk_index")
                if chunk_index is not None:
                    responses_dict[str(chunk_index)] = entry
        sample_result["responses"] = responses_dict
    elif not isinstance(responses, dict):
        sample_result["responses"] = {}

    records = sample_result.setdefault("records", {})
    if isinstance(records, list):
        records_dict: Dict[str, Dict] = {}
        for entry in records:
            if isinstance(entry, dict):
                qidx = entry.get("question_index")
                if qidx is not None:
                    records_dict[str(qidx)] = entry
        sample_result["records"] = records_dict
    elif not isinstance(records, dict):
        sample_result["records"] = {}

    return sample_result


def print_qa(qidx: int, question: str, expected: Optional[str], predicted: Optional[str]) -> None:
    print(f"[{qidx}] question: {question}")
    print(f"[{qidx}] expected_answer: {expected}")
    print(f"[{qidx}] predicted_answer: {predicted}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Mirix memory on LoCoMo.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/locomo10.json"),
        help="Path to LoCoMo dataset JSON.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to evaluate.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limit number of questions per sample.",
    )
    parser.add_argument(
        "--run-llm",
        action="store_true",
        default=True,
        help="Call the LLM to answer questions.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("locomo_run"),
        help=(
            "Output sub-folder name. The path is resolved relative to "
            "<repo>/evals/results/locomo/, so passing 'foo' writes to "
            "evals/results/locomo/foo. Absolute paths are still honored "
            "but warned about, since they bypass the locomo namespace."
        ),
    )
    parser.add_argument(
        "--mirix_config_path",
        type=Path,
        default=None,
        help="Path to Mirix config file.",
    )
    args = parser.parse_args()

    items = load_locomo(args.data)
    if args.limit is not None:
        items = items[: args.limit]

    mirix_client_id = os.environ.get("MIRIX_CLIENT_ID", "mirix-eval-client")
    mirix_org_id = os.environ.get("MIRIX_ORG_ID", "mirix-eval-org")

    # Force every main_eval run into the LoCoMo namespace so MAB and LoCoMo
    # outputs cannot bleed into each other. The user can still pass an
    # absolute path to break out (e.g. for one-off experiments), but a warning
    # makes the divergence explicit.
    locomo_root = Path(__file__).resolve().parent / "results" / "locomo"
    if args.output_path.is_absolute():
        print(
            f"[main_eval] WARNING: --output_path is absolute ({args.output_path}); "
            f"writing outside evals/results/locomo/ namespace.",
        )
        output_path = args.output_path
    else:
        output_path = locomo_root / args.output_path
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"[main_eval] writing per-sample results to {output_path}")

    # Server-side token tracker is always-on (see mirix/database/token_tracker.py).
    # We just need to (a) reset before each sample's ingest, (b) snapshot after
    # ingest to get "build" tokens, (c) snapshot after QA to get "query" tokens.
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
            for k in s: s[k] += v.get(k, 0)
        return s

    for item in items:
        sample_id = item.get("sample_id")
        if sample_id is None:
            continue
        sample_path = output_path / f"{sample_id}.json"

        task_agent = TaskAgent(mirix_config_path=str(args.mirix_config_path), client_id=mirix_client_id, org_id=mirix_org_id, user_id=sample_id) if args.run_llm else None

        sample_result = load_sample_result(sample_path)
        if sample_result is None:
            sample_result = {
                "sample_id": sample_id,
                "timings": {"add_chunk": {}, "wrap_user_prompt": {}, "answer": {}},
                "responses": {},
                "records": {},
            }

        sample_result.setdefault("sample_id", sample_id)
        sample_result = normalize_sample_result(sample_result)

        memory_system = MirixMemorySystem(user_id=sample_id,
                    mirix_config_path=str(args.mirix_config_path),
                    client=task_agent.mirix_client)

        # Reset server-side token counter so build_tokens reflects only this sample's ingest
        _reset_tokens()

        conversation = item.get("conversation", {})
        for idx, session in enumerate(iter_sessions(conversation), start=1):
            idx_key = str(idx)
            if idx_key in sample_result["responses"]:
                continue
            date_time_key = f"session_{idx}_date_time"
            date_time = conversation.get(date_time_key)
            if date_time is None:
                date_time = conversation.get(f"session_{idx + 1}_date_time")
            chunk = format_session_chunk(session, date_time=date_time)

            start = time.perf_counter()

            response = memory_system.add_chunk(chunk, raw_input=chunk)

            elapsed = time.perf_counter() - start

            sample_result["responses"][idx_key] = {
                "type": "add_chunk",
                "chunk_index": idx,
                "question_index": None,
                "response": response,
            }
            sample_result["timings"]["add_chunk"][idx_key] = elapsed
            save_sample_result(sample_path, sample_result)

        # Snapshot build tokens (everything since reset, before any QA runs)
        build_stats = _snapshot_tokens()
        sample_result["token_stats"] = {"build_raw": build_stats, "build_sum": _sum_tokens(build_stats)}
        save_sample_result(sample_path, sample_result)

        qa_list = item.get("qa", [])
        if args.max_questions is not None:
            qa_list = qa_list[: args.max_questions]

        for qidx, qa in enumerate(qa_list, start=1):
            qidx_key = str(qidx)
            if qidx_key in sample_result["records"]:
                record = sample_result["records"][qidx_key]
                print_qa(
                    qidx,
                    record.get("question", ""),
                    record.get("expected_answer"),
                    record.get("predicted_answer"),
                )
                continue
            question = qa.get("question", "")
            expected_answer = qa.get("answer")
            if expected_answer is None:
                record = {
                    "sample_id": sample_id,
                    "question_index": qidx,
                    "question": question,
                    "expected_answer": expected_answer,
                    "evidence": qa.get("evidence"),
                    "category": qa.get("category"),
                    "prompt": None,
                    "predicted_answer": None,
                    "messages": None,
                    "usage": None,
                    "usage_total": None,
                }
                sample_result["records"][qidx_key] = record
                save_sample_result(sample_path, sample_result)
                print_qa(qidx, question, expected_answer, None)
                continue
            start = time.perf_counter()
            input_messages = memory_system.wrap_user_prompt(question)
            sample_result["timings"]["wrap_user_prompt"][qidx_key] = (
                time.perf_counter() - start
            )
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
                sample_result["timings"]["answer"][qidx_key] = (
                    time.perf_counter() - start
                )

            record = {
                "sample_id": sample_id,
                "question_index": qidx,
                "question": question,
                "expected_answer": expected_answer,
                "evidence": qa.get("evidence"),
                "category": qa.get("category"),
                "input_messages": input_messages,
                "predicted_answer": predicted,
                "messages": message_trace,
                "usage": usage_trace,
                "usage_total": usage_total,
            }
            sample_result["records"][qidx_key] = record
            save_sample_result(sample_path, sample_result)
            print_qa(qidx, question, expected_answer, predicted)

        try:
            all_memories = memory_system.list_all_memories()
        except Exception as exc:
            all_memories = {
                "success": False,
                "error": str(exc),
                "user_id": sample_id,
            }

        memories_path = output_path / f"{sample_id}_memories.json"
        with memories_path.open("w", encoding="utf-8") as handle:
            json.dump(all_memories, handle, ensure_ascii=False, indent=2)

        # Snapshot post-QA total tokens. "query_tokens" is server-side retrieval
        # cost only (keyword extraction + LightRAG sub-calls). The actual QA
        # answer LLM call goes through task_agent (client-side OpenAI), tracked
        # separately in records[*].usage_total.
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
