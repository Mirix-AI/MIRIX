import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from mirix_memory_system import MirixMemorySystem
from task_agent import TaskAgent


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


def format_session_chunk(session: Dict) -> str:
    header = f"Session {session['number']}"
    if session.get("date_time"):
        header += f" ({session['date_time']})"
    lines = [header]
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
        default=Path("results"),
        help="Output folder for per-sample JSON results.",
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

    mirix_api_key = os.environ.get("MIRIX_API_KEY") or ""
    if not mirix_api_key:
        raise RuntimeError(
            "MIRIX_API_KEY must be set in the environment to run evals against the local server. "
            "Tip: run ./.venv/bin/python samples/generate_demo_api_key.py and export MIRIX_API_KEY to the printed value."
        )

    output_path = args.output_path
    output_path.mkdir(parents=True, exist_ok=True)

    for item in items:
        sample_id = item.get("sample_id")
        if sample_id is None:
            continue
        sample_path = output_path / f"{sample_id}.json"

        task_agent = TaskAgent(mirix_config_path=str(args.mirix_config_path), mirix_api_key=mirix_api_key, user_id=sample_id) if args.run_llm else None

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
                    mirix_api_key=mirix_api_key)

        conversation = item.get("conversation", {})
        for idx, session in enumerate(iter_sessions(conversation), start=1):
            idx_key = str(idx)
            if idx_key in sample_result["responses"]:
                continue
            chunk = format_session_chunk(session)
            date_time_key = f"session_{idx}_date_time"
            date_time = conversation.get(date_time_key)
            if date_time is None:
                date_time = conversation.get(f"session_{idx + 1}_date_time")
            
            # Timestamp context for memory agents
            timestamp_context = f"The conversation is timestamped at {date_time}.\n\n" if date_time else ""
            chunk_with_instruction = timestamp_context + chunk
            start = time.perf_counter()

            response = memory_system.add_chunk(chunk_with_instruction, raw_input=chunk)

            elapsed = time.perf_counter() - start

            sample_result["responses"][idx_key] = {
                "type": "add_chunk",
                "chunk_index": idx,
                "question_index": None,
                "response": response,
            }
            sample_result["timings"]["add_chunk"][idx_key] = elapsed
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
            all_memories = memory_system.list_all_memories(limit=0)
        except Exception as exc:
            all_memories = {
                "success": False,
                "error": str(exc),
                "user_id": sample_id,
            }

        memories_path = output_path / f"{sample_id}_memories.json"
        with memories_path.open("w", encoding="utf-8") as handle:
            json.dump(all_memories, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
